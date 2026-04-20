import sys

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <filename> <count>")
        sys.exit(1)
    
    filename = sys.argv[1]
    count = int(sys.argv[2])
    
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if i >= count:
                break
            word = line.strip().split(',')[0]
            print(word)

if __name__ == '__main__':
    main()
