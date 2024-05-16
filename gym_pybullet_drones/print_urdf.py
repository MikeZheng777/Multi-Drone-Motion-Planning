import pybullet_data
import os
import xml.etree.ElementTree as ET

# Set the data path for PyBullet
pybullet_data_path = pybullet_data.getDataPath()
print(pybullet_data_path)

# Function to extract robot name from a URDF file
# def get_robot_name_from_urdf(file_path):
#     try:
#         tree = ET.parse(file_path)
#         root = tree.getroot()
#         # Return the 'name' attribute of the <robot> tag
#         return root.attrib['name']
#     except ET.ParseError:
#         print(f"Error parsing {file_path}")
#         return None

# # List all URDF files and their robot names in the PyBullet data directory
# for root, dirs, files in os.walk(pybullet_data_path):
#     for file in files:
#         if file.endswith(".urdf"):
#             file_path = os.path.join(root, file)
#             robot_name = get_robot_name_from_urdf(file_path)
#             if robot_name:
#                 print(f"File: {file}, Robot Name: {robot_name}")

