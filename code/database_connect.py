import pyodbc
import numpy as np


# --------------------------------------------------------#
#       把东西放到数据库中
# --------------------------------------------------------#
# 连接到 SQL Server 数据库
def putDatabase(name, known_face_encodings):
    flat = 0
    conn = pyodbc.connect('DRIVER={SQL Server};SERVER=LEGION-Y9000P\SQLEXPRESS;DATABASE=SafeFace;UID=sa;PWD=')

    # 创建一个游标对象，用于执行 SQL 语句
    cursor = conn.cursor()
    try:
        # 将数据插入到数据库
        character_info_np = np.array(known_face_encodings, dtype=np.float64)
        character_info_bytes = character_info_np.tobytes()

        # 将数据插入到数据库
        cursor.execute("INSERT INTO [characters] ([name], [info]) VALUES (?, ?)", (name, character_info_bytes))
    except pyodbc.IntegrityError:
        flat = 1
        print("已注册")
        return flat
    # 提交更改
    conn.commit()
    # 关闭数据库连接
    conn.close()
    return flat


def putDatabaseAndTable(names, known_face_encodings):
    # 连接到 SQL Server 数据库
    conn = pyodbc.connect('DRIVER={SQL Server};SERVER=LEGION-Y9000P\SQLEXPRESS;DATABASE=SafeFace;UID=sa;PWD=')

    # 创建一个游标对象，用于执行 SQL 语句
    cursor = conn.cursor()

    # 检查表是否存在
    cursor.execute("SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'characters'")
    table_exists = cursor.fetchone() is not None

    # 如果表不存在，则创建表
    if not table_exists:
        # 创建表
        cursor.execute('''CREATE TABLE [characters]
                          ([name] VARCHAR(50) NOT NULL primary key,
                           [info] VARBINARY(MAX))''')

    # 将数据插入到数据库
    for i in range(len(names)):
        try:
            name = names[i]
            character_info = known_face_encodings[i]
            character_info_np = np.array(character_info, dtype=np.float64)
            character_info_bytes = character_info_np.tobytes()

            # 将数据插入到数据库
            cursor.execute("INSERT INTO [characters] ([name], [info]) VALUES (?, ?)", (name, character_info_bytes))

        except pyodbc.IntegrityError:
            pass

    # 提交更改
    conn.commit()

    # 关闭数据库连接
    conn.close()


def load_data():
    # 连接到 SQL Server 数据库(对应参数连接到不同数据库需要进行修改)
    conn = pyodbc.connect('DRIVER={SQL Server};SERVER=LEGION-Y9000P\SQLEXPRESS;DATABASE=SafeFace;UID=sa;PWD=')

    # 创建一个游标对象，用于执行 SQL 语句
    cursor = conn.cursor()
    cursor.execute("SELECT name, info from characters")

    # 获取查询结果
    rows = cursor.fetchall()

    # 结果保存在数组中
    face_names = []
    face_encodings = []

    for row in rows:
        name = row.name
        info_bytes = row.info
        matrix = np.frombuffer(info_bytes, dtype=np.float64)
        face_names.append(name)
        face_encodings.append(matrix)

    return [face_names, face_encodings]


if __name__ == '__main__':
    lst = load_data()
    for i in range(len(lst[0])):
        print(lst[0][i], lst[1][i])
