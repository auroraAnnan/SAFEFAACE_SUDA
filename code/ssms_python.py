import pyodbc
import numpy as np

# 连接到 SQL Server 数据库
conn = pyodbc.connect('DRIVER={SQL Server};SERVER=Legion-Y9000p\SQLEXPRESS;DATABASE=SafeFace;UID=sa;PWD=')

# 创建一个游标对象，用于执行 SQL 语句
cursor = conn.cursor()

# 检查表是否存在
cursor.execute("SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'characters'")
table_exists = cursor.fetchone() is not None

# 如果表不存在，则创建表
if not table_exists:
    # 创建表
    cursor.execute('''CREATE TABLE [characters]
                      ([name] VARCHAR(50) NOT NULL PRIMARY KEY,
                       [info] VARBINARY(MAX))''')

# 读取人物名称文件
names_data = np.load('E:\Python datas\SAFEFAACE_SUDA\code\model_data\mobilenet_names.npy')
names = names_data.tolist()

# 读取人物信息文件
info_data = np.load('E:\Python datas\SAFEFAACE_SUDA\code\model_data\mobilenet_face_encoding.npy')
info = info_data.tolist()

# 将数据插入到数据库
for i in range(len(names)):
    name = names[i]
    character_info = info[i]
    character_info_np = np.array(character_info, dtype=np.float64)
    character_info_bytes = character_info_np.tobytes()

    # 插入数据
    cursor.execute("INSERT INTO [characters] ([name], [info]) VALUES (?, ?)", (name, character_info_bytes))

# 提交更改
conn.commit()

# 关闭数据库连接
conn.close()