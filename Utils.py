from Settings import Settings
import subprocess

def execute_parser(project_path):
    depends_path = f"{Settings.DIRECTORY}\\depends\\target\\depends-0.9.7-package\\depends-0.9.7"
    command = "java -jar depends.jar "
    # output path
    command += f"-d {Settings.DIRECTORY}\\data\\depends\\{Settings.PROJECT_NAME} "
    # language
    command += "java "
    # src path
    command += f"{Settings.PROJECT_PATH} "
    # output file
    command += f"{Settings.PROJECT_NAME}.json --auto-include "


    print(f"Invoking parsing: {command}")
    subprocess.call(command, cwd=depends_path, shell=True)
