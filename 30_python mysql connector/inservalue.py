import mysql.connector

conn = mysql.connector.connect(host = 'localhost',user = 'root',password='0241',database = 'pythondb')

mycursor = conn.cursor()


sql = 'insert into student (name , branch ,id) values(%s,%s,%s)'

val = [('varnit','cse','56'),('parma','IT','78'),('subham','me','80')]

mycursor.executemany(sql,val)
conn.commit()
print(mycursor.rowcount,'record inserted')
