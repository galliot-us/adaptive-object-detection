from utils.parse_label_map import create_category_index_dict
import sys

def main(argv):
    label_path = argv[1]
    labels_dict = create_category_index_dict(label_path)
    f = open("/repo/deepstream-data/labels.txt", "w")
    for label_dict in labels_dict:
        f.write(labels_dict[label_dict].get("name")+'\n')
    f.close()

   
if __name__ == "__main__":
    main(sys.argv)
