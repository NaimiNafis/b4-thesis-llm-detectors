from libcst_rename.runner import process_file


if __name__ == '__main__':
    input_dataset_path = './dataset/python/test_no_comment.jsonl'
    output_dataset_path = './script/test_original_dataset_A_rename.jsonl'
    process_file(input_dataset_path, output_dataset_path)