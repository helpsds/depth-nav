import os
import re
import shutil  # 新增：用于文件复制
 
def get_numbered_images(folder_path):
    """获取文件夹中所有带数字编号的图片文件，并按编号排序"""
    # 支持的图片扩展名
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
 
    # 正则表达式匹配文件名中的数字（假设文件名格式为xxx数字xxx.扩展名）
    number_pattern = re.compile(r'(\d+)')
 
    image_files = []
 
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        # 检查是否为图片文件
        if os.path.isfile(file_path) and file.lower().endswith(image_extensions):
            # 提取文件名中的数字
            match = number_pattern.search(file)
            if match:
                try:
                    number = int(match.group(1))
                    image_files.append((file, number))
                except ValueError:
                    continue
 
    # 按数字编号排序
    image_files.sort(key=lambda x: x[1])
    # 返回排序后的文件名列表
    return [f[0] for f in image_files]
 
def copy_and_rename_images(parent_dir):
    """
    重命名指定文件夹中的所有子文件夹为traj_1, traj_2, ...
    并为每个子文件夹中的最后一个编号图片**复制副本**，重命名为T_num
    参数:
        parent_dir: 包含子文件夹的父文件夹路径
    """
    # 检查父文件夹是否存在
    if not os.path.exists(parent_dir):
        print(f"错误: 文件夹 '{parent_dir}' 不存在")
        return
 
    if not os.path.isdir(parent_dir):
        print(f"错误: '{parent_dir}' 不是一个文件夹")
        return
 
    # 获取所有子文件夹，排除文件
    subfolders = [f for f in os.listdir(parent_dir)
                  if os.path.isdir(os.path.join(parent_dir, f))]
 
    if not subfolders:
        print(f"在 '{parent_dir}' 中没有找到子文件夹")
        return
 
    # 按创建时间排序
    subfolders.sort(key=lambda x: os.path.getctime(os.path.join(parent_dir, x)))
 
    # 重命名子文件夹并处理图片（复制+重命名副本）
    for i, folder in enumerate(subfolders, start=1):
        old_folder_path = os.path.join(parent_dir, folder)
        new_folder_name = f"traj_{i}"
        new_folder_path = os.path.join(parent_dir, new_folder_name)
 
        # 如果新文件夹名称已存在，跳过
        if os.path.exists(new_folder_path):
            print(f"警告: 文件夹 '{new_folder_name}' 已存在，跳过重命名 '{folder}'")
            continue
 
        # 先重命名文件夹
        os.rename(old_folder_path, new_folder_path)
        print(f"重命名文件夹: '{folder}' -> '{new_folder_name}'")
 
        # 获取文件夹中的编号图片并排序
        image_files = get_numbered_images(new_folder_path)
 
        if image_files:
            # 获取最后一个图片（编号最大的图片）
            last_image = image_files[-1]
            original_image_path = os.path.join(new_folder_path, last_image)  # 原图片路径
 
            # 获取图片扩展名，构建副本名称（T_num.扩展名）
            _, ext = os.path.splitext(last_image)
            copied_image_name = f"T_{i}{ext}"
            copied_image_path = os.path.join(new_folder_path, copied_image_name)  # 副本路径
 
            # 如果副本已存在，跳过复制
            if os.path.exists(copied_image_path):
                print(f"警告: 副本 '{copied_image_name}' 已存在，跳过复制")
                continue
 
            # 复制原图片到副本（核心修改：用shutil.copy复制而非os.rename重命名）
            shutil.copy(original_image_path, copied_image_path)
            print(f"  已复制并命名: '{last_image}' -> '{copied_image_name}'（原图片保留）")
        else:
            print(f"  在 '{new_folder_name}' 中未找到带编号的图片")
 
    print(f"完成！共处理了 {len(subfolders)} 个子文件夹")
 
 
if __name__ == "__main__":
    # 替换为你的父文件夹路径
    parent_folder = "/home/yyz/data/go_stanford"
 
    copy_and_rename_images(parent_folder)