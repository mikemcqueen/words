#include <cctype>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "simdjson.h"

namespace py = pybind11;

namespace {

constexpr uint8_t kUnknownLabel = 0;
constexpr uint8_t kNoLabel = 1;
constexpr uint8_t kYesLabel = 2;

struct DirectionProjection {
  std::string name;
  uint8_t label = kUnknownLabel;
  double prob = 0.0;
};

struct ProjectedRow {
  std::string pair;
  std::vector<DirectionProjection> directions;
};

std::string parse_key_from_path(const std::string &path) {
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

uint8_t classify_token(const std::string &token) {
  std::string normalized = token;
  for (char &ch : normalized) {
    ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
  }
  if (normalized.rfind("YES", 0) == 0) {
    return kYesLabel;
  }
  if (normalized.rfind("NO", 0) == 0) {
    return kNoLabel;
  }
  return kUnknownLabel;
}

DirectionProjection project_direction_value(simdjson::dom::element value, const std::string &path) {
  DirectionProjection projection;

  auto arr_result = value.get_array();
  if (!arr_result.error()) {
    for (auto token_entry : arr_result.value_unsafe()) {
      auto token_obj_result = token_entry.get_object();
      if (token_obj_result.error()) {
        throw std::runtime_error("invalid token-prob entry in " + path);
      }
      for (auto token_field : token_obj_result.value_unsafe()) {
        auto prob_result = token_field.value.get_double();
        if (prob_result.error()) {
          throw std::runtime_error("invalid probability in " + path);
        }
        projection.label = classify_token(std::string(token_field.key));
        projection.prob = prob_result.value_unsafe();
        return projection;
      }
    }
    return projection;
  }

  auto obj_result = value.get_object();
  if (!obj_result.error()) {
    for (auto token_field : obj_result.value_unsafe()) {
      auto prob_result = token_field.value.get_double();
      if (prob_result.error()) {
        throw std::runtime_error("invalid probability in " + path);
      }
      projection.label = classify_token(std::string(token_field.key));
      projection.prob = prob_result.value_unsafe();
      return projection;
    }
    return projection;
  }

  auto scalar_result = value.get_double();
  if (scalar_result.error()) {
    throw std::runtime_error("invalid logprobs entry in " + path);
  }
  projection.prob = scalar_result.value_unsafe();
  return projection;
}

template <typename RowT>
void validate_pair_alignment(const std::vector<std::string> &paths,
                             const std::vector<RowT> &primary,
                             const std::vector<RowT> &secondary,
                             size_t secondary_index) {
  const size_t common = std::min(primary.size(), secondary.size());
  for (size_t j = 0; j < common; ++j) {
    if (primary[j].pair != secondary[j].pair) {
      throw std::runtime_error("pair mismatch between " + paths[0] + " and " + paths[secondary_index] +
                               ": differing pair " + secondary[j].pair);
    }
  }
}

class ProjectedChunk {
 public:
  ProjectedChunk(std::vector<std::string> keys, std::vector<std::string> directions, size_t rows,
                 size_t capacity, std::vector<uint8_t> labels, std::vector<double> probs,
                 std::vector<std::string> pairs)
      : keys_(std::move(keys)),
        directions_(std::move(directions)),
        rows_(rows),
        capacity_(capacity),
        labels_(std::move(labels)),
        probs_(std::move(probs)),
        pairs_(std::move(pairs)) {}

  const std::vector<std::string> &keys() const { return keys_; }
  const std::vector<std::string> &directions() const { return directions_; }
  size_t size() const { return rows_; }

  py::array_t<uint8_t> labels() const {
    return py::array_t<uint8_t>(
        {static_cast<py::ssize_t>(keys_.size()), static_cast<py::ssize_t>(rows_),
         static_cast<py::ssize_t>(directions_.size())},
        {static_cast<py::ssize_t>(capacity_ * directions_.size() * sizeof(uint8_t)),
         static_cast<py::ssize_t>(directions_.size() * sizeof(uint8_t)),
         static_cast<py::ssize_t>(sizeof(uint8_t))},
        labels_.data(), py::cast(this));
  }

  const std::vector<std::string> &pairs() const { return pairs_; }

  py::array_t<double> probs() const {
    return py::array_t<double>(
        {static_cast<py::ssize_t>(keys_.size()), static_cast<py::ssize_t>(rows_),
         static_cast<py::ssize_t>(directions_.size())},
        {static_cast<py::ssize_t>(capacity_ * directions_.size() * sizeof(double)),
         static_cast<py::ssize_t>(directions_.size() * sizeof(double)),
         static_cast<py::ssize_t>(sizeof(double))},
        probs_.data(), py::cast(this));
  }

 private:
  std::vector<std::string> keys_;
  std::vector<std::string> directions_;
  size_t rows_;
  size_t capacity_;
  std::vector<uint8_t> labels_;
  std::vector<double> probs_;
  std::vector<std::string> pairs_;
};

class ProjectedAlignedJsonlReader {
 public:
  ProjectedAlignedJsonlReader(const std::vector<std::string> &paths, size_t chunk_size)
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

  ProjectedAlignedJsonlReader(const ProjectedAlignedJsonlReader &) = delete;
  ProjectedAlignedJsonlReader &operator=(const ProjectedAlignedJsonlReader &) = delete;
  ProjectedAlignedJsonlReader(ProjectedAlignedJsonlReader &&) = default;
  ProjectedAlignedJsonlReader &operator=(ProjectedAlignedJsonlReader &&) = default;

  ProjectedAlignedJsonlReader &iter() { return *this; }

  ProjectedChunk next() {
    ProjectedChunk chunk({}, {}, 0, 0, {}, {}, {});
    {
      py::gil_scoped_release release;
      chunk = read_next_chunk();
    }
    if (chunk.size() == 0) {
      throw py::stop_iteration();
    }
    return chunk;
  }

 private:
  ProjectedChunk read_next_chunk() {
    std::vector<ProjectedRow> primary = read_rows_for_file(0, chunk_size_);
    if (primary.empty()) {
      return ProjectedChunk(keys_, directions_, 0, 0, {}, {}, {});
    }

    if (directions_.empty()) {
      if (primary[0].directions.empty()) {
        throw std::runtime_error("missing directions in " + paths_[0]);
      }
      directions_.reserve(primary[0].directions.size());
      for (const auto &direction : primary[0].directions) {
        directions_.push_back(direction.name);
      }
    }
    validate_direction_schema(primary, 0);

    const size_t n_files = paths_.size();
    const size_t n_rows = primary.size();
    const size_t n_dirs = directions_.size();
    size_t live_rows = n_rows;
    std::vector<std::pair<size_t, size_t>> truncations;

    std::vector<uint8_t> labels(n_files * n_rows * n_dirs, kUnknownLabel);
    std::vector<double> probs(n_files * n_rows * n_dirs, 0.0);

    fill_arrays(primary, 0, n_rows, labels, probs);

    for (size_t file_index = 1; file_index < n_files; ++file_index) {
      std::vector<ProjectedRow> rows = read_rows_for_file(file_index, n_rows);
      if (rows.size() < live_rows) {
        truncations.emplace_back(file_index, live_rows - rows.size());
        live_rows = rows.size();
      }
      validate_pair_alignment(paths_, primary, rows, file_index);
      validate_direction_schema(rows, file_index);
      fill_arrays(rows, file_index, n_rows, labels, probs);
    }

    if (!truncations.empty() && live_rows > 0) {
      for (const auto &t : truncations) {
        std::fprintf(stderr, "WARNING: %s truncated; dropped %zu pair(s) from chunk\n",
                     paths_[t.first].c_str(), t.second);
      }
    }

    std::vector<std::string> pair_ids;
    pair_ids.reserve(live_rows);
    for (size_t i = 0; i < live_rows; ++i) {
      pair_ids.push_back(primary[i].pair);
    }
    return ProjectedChunk(keys_, directions_, live_rows, n_rows, std::move(labels), std::move(probs), std::move(pair_ids));
  }

  void fill_arrays(const std::vector<ProjectedRow> &rows, size_t file_index, size_t capacity,
                   std::vector<uint8_t> &labels, std::vector<double> &probs) const {
    const size_t n_dirs = directions_.size();
    const size_t file_offset = file_index * capacity * n_dirs;
    for (size_t row_index = 0; row_index < rows.size(); ++row_index) {
      for (size_t dir_index = 0; dir_index < n_dirs; ++dir_index) {
        const size_t offset = file_offset + row_index * n_dirs + dir_index;
        labels[offset] = rows[row_index].directions[dir_index].label;
        probs[offset] = rows[row_index].directions[dir_index].prob;
      }
    }
  }

  void validate_direction_schema(const std::vector<ProjectedRow> &rows, size_t file_index) const {
    for (const auto &row : rows) {
      if (row.directions.size() != directions_.size()) {
        throw std::runtime_error("direction mismatch in " + paths_[file_index] + ": differing direction count");
      }
      for (size_t dir_index = 0; dir_index < directions_.size(); ++dir_index) {
        if (row.directions[dir_index].name != directions_[dir_index]) {
          throw std::runtime_error("direction mismatch between " + paths_[0] + " and " + paths_[file_index] +
                                   ": differing direction " + row.directions[dir_index].name);
        }
      }
    }
  }

  std::vector<ProjectedRow> read_rows_for_file(size_t file_index, size_t count) {
    std::vector<ProjectedRow> rows;
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

  ProjectedRow parse_line(size_t file_index, const std::string &line) {
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

    ProjectedRow row;
    row.pair = std::string(pair_result.value_unsafe());

    auto logprobs_result = doc["logprobs"].get_object();
    if (logprobs_result.error()) {
      throw std::runtime_error("missing logprobs in " + paths_[file_index]);
    }

    for (auto field : logprobs_result.value_unsafe()) {
      DirectionProjection projection = project_direction_value(field.value, paths_[file_index]);
      projection.name = std::string(field.key);
      row.directions.push_back(std::move(projection));
    }

    return row;
  }

  std::vector<std::string> paths_;
  size_t chunk_size_;
  std::vector<std::string> keys_;
  std::vector<std::string> directions_;
  std::vector<std::ifstream> handles_;
  std::vector<simdjson::dom::parser> parsers_;
};

}  // namespace

PYBIND11_MODULE(_compare_native, m) {
  m.attr("LABEL_UNKNOWN") = py::int_(kUnknownLabel);
  m.attr("LABEL_NO") = py::int_(kNoLabel);
  m.attr("LABEL_YES") = py::int_(kYesLabel);

  py::class_<ProjectedChunk>(m, "ProjectedChunk")
      .def("keys", &ProjectedChunk::keys)
      .def("directions", &ProjectedChunk::directions)
      .def("labels", &ProjectedChunk::labels)
      .def("probs", &ProjectedChunk::probs)
      .def("pairs", &ProjectedChunk::pairs)
      .def_property_readonly("size", &ProjectedChunk::size);

  py::class_<ProjectedAlignedJsonlReader>(m, "ProjectedAlignedJsonlReader")
      .def(py::init<const std::vector<std::string> &, size_t>(), py::arg("paths"), py::arg("chunk_size"))
      .def("__iter__", [](ProjectedAlignedJsonlReader &self) -> ProjectedAlignedJsonlReader & { return self; },
           py::return_value_policy::reference_internal)
      .def("__next__", &ProjectedAlignedJsonlReader::next);
}
