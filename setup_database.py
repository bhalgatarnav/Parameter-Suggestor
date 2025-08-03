import os
import sys
import mysql.connector
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_database():
    """Set up MySQL database and user"""
    # Load environment variables
    load_dotenv()
    
    # Get database configuration
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_USER = os.getenv('DB_USER', 'root')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    DB_NAME = os.getenv('DB_NAME', 'material_library')
    
    try:
        # Connect to MySQL server as root
        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = connection.cursor()
        
        # Create database if it doesn't exist
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
        logger.info(f"Database '{DB_NAME}' created or already exists")
        
        # Use the database
        cursor.execute(f"USE {DB_NAME}")
        
        # Grant privileges to the user
        grant_stmt = f"GRANT ALL PRIVILEGES ON {DB_NAME}.* TO '{DB_USER}'@'localhost' IDENTIFIED BY '{DB_PASSWORD}'"
        cursor.execute(grant_stmt)
        cursor.execute("FLUSH PRIVILEGES")
        logger.info(f"Granted privileges to user '{DB_USER}'")
        
        # Create necessary tables
        create_tables(cursor)
        
        connection.commit()
        logger.info("Database setup completed successfully!")
        
    except mysql.connector.Error as err:
        logger.error(f"MySQL Error: {err}")
        if err.errno == 1045:  # Access denied error
            logger.error("Please check your MySQL root password in .env file")
            logger.error("Make sure you can connect to MySQL using:")
            logger.error(f"  mysql -u {DB_USER} -p")
        elif err.errno == 1698:  # Access denied for root@localhost
            logger.error("Please set a root password for MySQL:")
            logger.error("1. Connect to MySQL:")
            logger.error("   mysql -u root")
            logger.error("2. Set new password:")
            logger.error("   ALTER USER 'root'@'localhost' IDENTIFIED BY 'your_password';")
            logger.error("3. Update DB_PASSWORD in .env file")
        sys.exit(1)
        
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()

def create_tables(cursor):
    """Create necessary database tables"""
    
    # Materials table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS materials (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        description TEXT,
        metalness FLOAT,
        roughness FLOAT,
        color_r FLOAT,
        color_g FLOAT,
        color_b FLOAT,
        specular_r FLOAT,
        specular_g FLOAT,
        specular_b FLOAT,
        embedding JSON,
        visual_embedding JSON,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    )
    """)
    
    # Create other necessary tables...
    logger.info("Database tables created successfully")

if __name__ == "__main__":
    setup_database() 