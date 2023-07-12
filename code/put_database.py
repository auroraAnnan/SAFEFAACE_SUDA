import pyodbc
import numpy as np
# --------------------------------------------------------#
#       把东西放到数据库中
# --------------------------------------------------------#
# 连接到 SQL Server 数据库
def putDatabase(name, known_face_encodings):
    flat = 0
    conn = pyodbc.connect('DRIVER={SQL Server};SERVER=.;DATABASE=users;UID=sa;PWD=1234')

    # 创建一个游标对象，用于执行 SQL 语句
    cursor = conn.cursor()
    try:
        # 将数据插入到数据库
        character_info_np = np.array(known_face_encodings)
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