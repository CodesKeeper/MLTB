# import pymysql
#
# pymysql.install_as_MySQLdb()
#
# # 打开数据库连接
# db = pymysql.connect("localhost", "lsy", "qqqwww", "mldb")
#
# # 使用cursor()方法获取操作游标
# cursor = db.cursor()
#
# # 使用execute()方法执行SQL语句
# cursor.execute("SELECT VERSION()")
#
# # 使用 fetchone()方法获取一条数据库。
# data = cursor.fetchone()
#
# print("Database version : %s " % data)
#
# # 关闭数据库
# db.close()
