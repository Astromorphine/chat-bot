import os
import sys

def generate_directory_tree(root_path, ignore_dirs=None, max_depth=None):
    """
    Генерирует древовидную структуру директорий и файлов
    :param root_path: Путь к корневой директории
    :param ignore_dirs: Список директорий для игнорирования
    :param max_depth: Максимальная глубина рекурсии
    :return: Многострочная строка с древовидной структурой
    """
    if ignore_dirs is None:
        ignore_dirs = {'__pycache__', '.git', '.idea', 'venv', 'env', '.vscode'}
    
    prefix = []
    output = []
    
    def add_line(name, is_dir=False, is_last=False):
        line = ''.join(prefix[:-1])
        line += '└── ' if is_last else '├── '
        line += name
        if is_dir:
            line += '/'
        output.append(line)
    
    def scan(current_path, depth=0):
        try:
            entries = os.listdir(current_path)
        except PermissionError:
            return
            
        # Фильтрация и сортировка
        valid_entries = []
        for name in entries:
            if name.startswith('.'):
                continue
            full_path = os.path.join(current_path, name)
            if os.path.isdir(full_path) and name in ignore_dirs:
                continue
            valid_entries.append((name, os.path.isdir(full_path)))
        
        # Сортировка: сначала директории, затем файлы
        valid_entries.sort(key=lambda x: (not x[1], x[0]))
        
        for i, (name, is_dir) in enumerate(valid_entries):
            is_last = (i == len(valid_entries) - 1)
            add_line(name, is_dir, is_last)
            
            if is_dir and (max_depth is None or depth < max_depth):
                full_path = os.path.join(current_path, name)
                
                # Обновление префикса
                prefix.append('    ' if is_last else '│   ')
                scan(full_path, depth + 1)
                prefix.pop()
    
    # Добавляем корневую директорию
    root_name = os.path.basename(root_path.rstrip(os.sep)) or os.path.basename(os.path.dirname(root_path))
    output.append(f"{root_name}/")
    
    # Сканируем корневую директорию
    scan(root_path)
    return '\n'.join(output)

if __name__ == "__main__":
    # Определяем путь (текущая директория или первый аргумент)
    path = '.' if len(sys.argv) < 2 else sys.argv[1]
    abs_path = os.path.abspath(path)
    
    if not os.path.exists(abs_path):
        print(f"Ошибка: Путь '{abs_path}' не существует")
        sys.exit(1)
        
    if not os.path.isdir(abs_path):
        print(f"Ошибка: '{abs_path}' не является директорией")
        sys.exit(1)
    
    # Генерируем и выводим структуру
    tree = generate_directory_tree(abs_path, max_depth=4)
    print(tree)