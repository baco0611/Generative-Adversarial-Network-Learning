import os

def rename_files_in_directory(directory):
    # Lấy danh sách các tệp trong thư mục
    files = os.listdir(directory)
    # Sắp xếp danh sách các tệp
    files.sort()
    
    for index, filename in enumerate(files):
        # Lấy đường dẫn đầy đủ của tệp
        old_path = os.path.join(directory, filename)
        # Kiểm tra xem đây có phải là một tệp hay không (bỏ qua thư mục)
        if os.path.isfile(old_path):
            # Lấy phần mở rộng của tệp
            file_extension = os.path.splitext(filename)[1]
            # Tạo tên mới cho tệp
            new_filename = f"file_{index + 1}{file_extension}"
            new_path = os.path.join(directory, new_filename)
            # Đổi tên tệp
            os.rename(old_path, new_path)
            print(f"Đổi tên: {old_path} -> {new_path}")

# Thư mục chứa các tệp cần đổi tên
directory_path = "./dataset"
rename_files_in_directory(directory_path)