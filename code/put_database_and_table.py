import pyodbc
import numpy as np

from interface import database_interface


def put_database_and_table(name, known_face_encodings):
    # 连接到 SQL Server 数据库
    conn = pyodbc.connect(database_interface('{SQL Server}',
                                             r'LEGION-Y9000P\SQLEXPRESS', 'SafeFace', 'sa', ''))

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
    character_info_np = np.array(known_face_encodings, dtype=np.float64)
    character_info_bytes = character_info_np.tobytes()

    # 将数据插入到数据库
    cursor.execute("INSERT INTO [characters] ([name], [info]) VALUES (?, ?)", (name, character_info_bytes))

    # 提交更改
    conn.commit()

    # 关闭数据库连接
    conn.close()
