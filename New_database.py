import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",            
  password="Material1100" 
)

mycursor = mydb.cursor()

mycursor.execute("CREATE DATABASE materials_library_db")

print(" Database 'materials_library_db' created successfully.")