#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "simdjson.h"

namespace py = pybind11;

struct TokenProb {
  std::string token;
  double prob;
};

struct DirectionValue {
  enum class Kind { kArray, kObject, kScalarDouble };
  Kind kind = Kind::kArray;
  std::vector<TokenProb> token_probs;
  double scalar_double = 0.0;
};

using DirectionLogprobs = std::vector<std::pair<std::string, DirectionValue>>;

struct ParsedRow {
  std::string pair;
  DirectionLogprobs logprobs;
};

class AlignedChunk {
 public:
  AlignedChunk(std::vector<std::string> keys, std::vector<std::vector<ParsedRow>> rows_by_file)
      : keys_(std::move(keys)), rows_by_file_(std::move(rows_by_file)) {}

  const std::vector<std::string> &keys() const { return keys_; }

  size_t size() const { return rows_by_file_.empty() ? 0 : rows_by_file_[0].size(); }

  std::vector<std::string> pairs() const {
    std::vector<std::string> result;
    if (rows_by_file_.empty()) {
      return result;
    }
    result.reserve(rows_by_file_[0].size());
    for (const auto &row : rows_by_file_[0]) {
      result.push_back(row.pair);
    }
    return result;
  }

  py::list rows_for_file(size_t index) const {
    if (index >= rows_by_file_.size()) {
      throw py::index_error("file index out of range");
    }
    py::list rows;
    for (const auto &row : rows_by_file_[index]) {
      py::dict py_row;
      py_row["pair"] = row.pair;
      py::dict py_logprobs;
      for (const auto &dir_entry : row.logprobs) {
        if (dir_entry.second.kind == DirectionValue::Kind::kArray) {
          py::list tokens;
          for (const auto &token_prob : dir_entry.second.token_probs) {
            py::dict token_entry;
            token_entry[py::str(token_prob.token)] = token_prob.prob;
            tokens.append(token_entry);
          }
          py_logprobs[py::str(dir_entry.first)] = tokens;
        } else if (dir_entry.second.kind == DirectionValue::Kind::kObject) {
          py::dict tokens;
          for (const auto &token_prob : dir_entry.second.token_probs) {
            tokens[py::str(token_prob.token)] = token_prob.prob;
          }
          py_logprobs[py::str(dir_entry.first)] = tokens;
        } else {
          py_logprobs[py::str(dir_entry.first)] = py::float_(dir_entry.second.scalar_double);
        }
      }
      py_row["logprobs"] = py_logprobs;
      rows.append(py_row);
    }
    return rows;
  }

 private:
  std::vector<std::string> keys_;
  std::vector<std::vector<ParsedRow>> rows_by_file_;
};

class AlignedJsonlReader {
 public:
  AlignedJsonlReader(const std::vector<std::string> &paths, size_t chunk_size)
      : paths_(paths), chunk_size_(chunk_size) {
    if (paths_.empty()) {
      throw std::runtime_error("at least one path is required");
    }
    if (chunk_size_ == 0) {
      throw std::runtime_error("chunk_size must be > 0");
    }

    keys_.reserve(paths_.size());
    handles_.reserve(paths_.size());
    parsers_.resize(paths_.size());
    for (const auto &path : paths_) {
      handles_.emplace_back(path);
      if (!handles_.back().is_open()) {
        throw std::runtime_error("failed to open " + path);
      }
      keys_.push_back(parse_key_from_path(path));
    }
  }

  AlignedJsonlReader(const AlignedJsonlReader &) = delete;
  AlignedJsonlReader &operator=(const AlignedJsonlReader &) = delete;
  AlignedJsonlReader(AlignedJsonlReader &&) = default;
  AlignedJsonlReader &operator=(AlignedJsonlReader &&) = default;

  AlignedJsonlReader &iter() { return *this; }

  AlignedChunk next() {
    std::vector<std::vector<ParsedRow>> rows_by_file;
    {
      py::gil_scoped_release release;
      rows_by_file = read_next_chunk();
    }
    if (rows_by_file.empty()) {
      throw py::stop_iteration();
    }
    return AlignedChunk(keys_, std::move(rows_by_file));
  }

 private:
  static std::string parse_key_from_path(const std::string &path) {
    const auto slash = path.find_last_of("/\\");
    const std::string filename = slash == std::string::npos ? path : path.substr(slash + 1);
    std::string stem = filename;
    const auto dot = stem.rfind('.');
    if (dot != std::string::npos) {
      stem = stem.substr(0, dot);
    }

    std::vector<std::string> parts;
    size_t start = 0;
    while (start <= stem.size()) {
      const auto pos = stem.find('_', start);
      if (pos == std::string::npos) {
        parts.push_back(stem.substr(start));
        break;
      }
      parts.push_back(stem.substr(start, pos - start));
      start = pos + 1;
    }
    if (parts.size() < 4) {
      throw std::runtime_error("could not parse key from filename: " + filename);
    }

    const std::regex prompt_id_re("^p\\d+$");
    int prompt_id_index = -1;
    for (int i = static_cast<int>(parts.size()) - 2; i >= 1; --i) {
      if (std::regex_match(parts[i], prompt_id_re)) {
        prompt_id_index = i;
        break;
      }
    }
    if (prompt_id_index <= 0 || prompt_id_index + 1 >= static_cast<int>(parts.size())) {
      throw std::runtime_error("could not parse key from filename: " + filename);
    }

    const std::string &prompt_file = parts[prompt_id_index - 1];
    const std::string &prompt_id = parts[prompt_id_index];
    std::string host;
    for (size_t i = static_cast<size_t>(prompt_id_index + 1); i < parts.size(); ++i) {
      if (!host.empty()) {
        host += "_";
      }
      host += parts[i];
    }
    std::string tag;
    const auto host_dot = host.find('.');
    if (host_dot != std::string::npos) {
      tag = host.substr(host_dot + 1);
    }
    return tag.empty() ? prompt_file + "." + prompt_id : prompt_file + "." + prompt_id + "." + tag;
  }

  std::vector<std::vector<ParsedRow>> read_next_chunk() {
    std::vector<ParsedRow> primary = read_rows_for_file(0, chunk_size_);
    if (primary.empty()) {
      return {};
    }

    std::vector<std::vector<ParsedRow>> rows_by_file(paths_.size());
    rows_by_file[0] = std::move(primary);
    const auto expected_rows = rows_by_file[0].size();

    for (size_t i = 1; i < paths_.size(); ++i) {
      rows_by_file[i] = read_rows_for_file(i, expected_rows);
      if (rows_by_file[i].size() != expected_rows) {
        throw std::runtime_error("pair mismatch between " + paths_[0] + " and " + paths_[i] +
                                 ": truncated secondary file");
      }
      for (size_t j = 0; j < expected_rows; ++j) {
        if (rows_by_file[0][j].pair != rows_by_file[i][j].pair) {
          throw std::runtime_error("pair mismatch between " + paths_[0] + " and " + paths_[i] +
                                   ": differing pair " + rows_by_file[i][j].pair);
        }
      }
    }

    return rows_by_file;
  }

  std::vector<ParsedRow> read_rows_for_file(size_t file_index, size_t count) {
    std::vector<ParsedRow> rows;
    rows.reserve(count);

    std::string line;
    while (rows.size() < count && std::getline(handles_[file_index], line)) {
      if (!line.empty() && line.back() == '\r') {
        line.pop_back();
      }
      if (line.empty()) {
        continue;
      }
      rows.push_back(parse_line(file_index, line));
    }
    return rows;
  }

  ParsedRow parse_line(size_t file_index, const std::string &line) {
    simdjson::padded_string padded(line);
    auto doc_result = parsers_[file_index].parse(padded);
    if (doc_result.error()) {
      throw std::runtime_error("invalid JSON in " + paths_[file_index] + ": " +
                               std::string(simdjson::error_message(doc_result.error())));
    }
    simdjson::dom::element doc = doc_result.value_unsafe();

    auto pair_result = doc["pair"].get_string();
    if (pair_result.error()) {
      throw std::runtime_error("missing pair in " + paths_[file_index]);
    }

    ParsedRow row;
    row.pair = std::string(pair_result.value_unsafe());

    auto logprobs_result = doc["logprobs"].get_object();
    if (logprobs_result.error()) {
      throw std::runtime_error("missing logprobs in " + paths_[file_index]);
    }

    for (auto field : logprobs_result.value_unsafe()) {
      const auto key = std::string(field.key);
      DirectionValue dir_value;
      auto arr_result = field.value.get_array();
      if (!arr_result.error()) {
        dir_value.kind = DirectionValue::Kind::kArray;
        for (auto token_entry : arr_result.value_unsafe()) {
          auto token_obj_result = token_entry.get_object();
          if (token_obj_result.error()) {
            throw std::runtime_error("invalid token-prob entry in " + paths_[file_index]);
          }
          for (auto token_field : token_obj_result.value_unsafe()) {
            auto prob_result = token_field.value.get_double();
            if (prob_result.error()) {
              throw std::runtime_error("invalid probability in " + paths_[file_index]);
            }
            dir_value.token_probs.push_back(
                TokenProb{std::string(token_field.key), prob_result.value_unsafe()});
          }
        }
      } else {
        auto obj_result = field.value.get_object();
        if (!obj_result.error()) {
          dir_value.kind = DirectionValue::Kind::kObject;
          for (auto token_field : obj_result.value_unsafe()) {
            auto prob_result = token_field.value.get_double();
            if (prob_result.error()) {
              throw std::runtime_error("invalid probability in " + paths_[file_index]);
            }
            dir_value.token_probs.push_back(
                TokenProb{std::string(token_field.key), prob_result.value_unsafe()});
          }
        } else {
          auto scalar_result = field.value.get_double();
          if (scalar_result.error()) {
            throw std::runtime_error("invalid logprobs entry in " + paths_[file_index]);
          }
          dir_value.kind = DirectionValue::Kind::kScalarDouble;
          dir_value.scalar_double = scalar_result.value_unsafe();
        }
      }
      row.logprobs.push_back({key, std::move(dir_value)});
    }

    return row;
  }

  std::vector<std::string> paths_;
  size_t chunk_size_;
  std::vector<std::string> keys_;
  std::vector<std::ifstream> handles_;
  std::vector<simdjson::dom::parser> parsers_;
};

PYBIND11_MODULE(_compare_native, m) {
  py::class_<AlignedChunk>(m, "AlignedChunk")
      .def("keys", &AlignedChunk::keys)
      .def("pairs", &AlignedChunk::pairs)
      .def("rows_for_file", &AlignedChunk::rows_for_file)
      .def_property_readonly("size", &AlignedChunk::size);

  py::class_<AlignedJsonlReader>(m, "AlignedJsonlReader")
      .def(py::init<const std::vector<std::string> &, size_t>(), py::arg("paths"), py::arg("chunk_size"))
      .def("__iter__", [](AlignedJsonlReader &self) -> AlignedJsonlReader & { return self; },
           py::return_value_policy::reference_internal)
      .def("__next__", &AlignedJsonlReader::next);
}
