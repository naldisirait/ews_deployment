import subprocess
import pkg_resources

def check_requirements(requirements_file):
    # Read the requirements from the file
    with open(requirements_file, 'r') as file:
        requirements = file.readlines()

    # Clean up each line to remove any extra whitespace
    requirements = [req.strip() for req in requirements]

    for requirement in requirements:
        try:
            # Check if the requirement is installed
            pkg_resources.require(requirement)
            print(f"{requirement} is already installed.")
        except pkg_resources.DistributionNotFound:
            print(f"{requirement} is NOT installed.")
        except pkg_resources.VersionConflict as e:
            print(f"{requirement} has a version conflict: {e}")

if __name__ == "__main__":
    # Path to the requirements.txt file
    requirements_file = "requirements.txt"
    
    check_requirements(requirements_file)
