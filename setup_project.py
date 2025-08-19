#!/usr/bin/env python3
"""
Project initialization script for Accelerating Materials Discovery RAG system.

This script sets up the development environment, creates necessary directories,
and provides guidance for getting started with the project.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import shutil


def print_header(text):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f" {text}")
    print(f"{'='*60}")


def print_step(step_num, text):
    """Print a formatted step."""
    print(f"\n[{step_num}] {text}")


def run_command(command, capture_output=True):
    """Run a shell command and return the result."""
    try:
        if isinstance(command, str):
            command = command.split()
        
        result = subprocess.run(
            command, 
            capture_output=capture_output, 
            text=True, 
            check=True
        )
        return result.stdout.strip() if capture_output else ""
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(command)}")
        print(f"Error: {e}")
        return None


def check_python_version():
    """Check if Python version is compatible."""
    print_step(1, "Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected.")
        print("❌ This project requires Python 3.8 or higher.")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected.")
    return True


def create_virtual_environment():
    """Create and activate virtual environment."""
    print_step(2, "Setting up virtual environment...")
    
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("🔄 Virtual environment already exists. Removing...")
        shutil.rmtree(venv_path)
    
    # Create virtual environment
    result = run_command([sys.executable, "-m", "venv", "venv"])
    if result is None:
        print("❌ Failed to create virtual environment.")
        return False
    
    print("✅ Virtual environment created successfully.")
    
    # Provide activation instructions
    if platform.system() == "Windows":
        activation_cmd = "venv\\Scripts\\activate"
    else:
        activation_cmd = "source venv/bin/activate"
    
    print(f"📝 To activate the virtual environment, run:")
    print(f"   {activation_cmd}")
    
    return True


def install_dependencies():
    """Install Python dependencies."""
    print_step(3, "Installing dependencies...")
    
    # Determine pip executable
    if platform.system() == "Windows":
        pip_cmd = "venv\\Scripts\\pip"
    else:
        pip_cmd = "venv/bin/pip"
    
    if not Path(pip_cmd).exists():
        print("❌ Virtual environment not found. Please create it first.")
        return False
    
    # Upgrade pip
    print("🔄 Upgrading pip...")
    run_command([pip_cmd, "install", "--upgrade", "pip"], capture_output=False)
    
    # Install requirements
    print("🔄 Installing requirements...")
    result = run_command([pip_cmd, "install", "-r", "requirements.txt"], capture_output=False)
    
    if result is None:
        print("❌ Failed to install some dependencies.")
        print("📝 You may need to install some dependencies manually.")
        return False
    
    print("✅ Dependencies installed successfully.")
    return True


def create_directories():
    """Create necessary project directories."""
    print_step(4, "Creating project directories...")
    
    directories = [
        "data/raw",
        "data/processed", 
        "data/interim",
        "data/external",
        "models/cache",
        "models/trained",
        "logs",
        "cache",
        "docs",
        "tests/test_data",
        "tests/test_output"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")


def setup_configuration():
    """Set up configuration files."""
    print_step(5, "Setting up configuration...")
    
    # Copy environment file
    env_example = Path("config/.env.example")
    env_file = Path("config/.env")
    
    if env_example.exists() and not env_file.exists():
        shutil.copy(env_example, env_file)
        print("✅ Created .env file from template.")
        print("📝 Please edit config/.env with your actual configuration values.")
    else:
        print("🔄 Configuration file already exists or template not found.")


def setup_git():
    """Initialize git repository and create branch."""
    print_step(6, "Setting up Git repository...")
    
    # Check if git is available
    git_version = run_command(["git", "--version"])
    if git_version is None:
        print("❌ Git not found. Please install Git to continue.")
        return False
    
    print(f"✅ {git_version}")
    
    # Initialize repository if needed
    if not Path(".git").exists():
        print("🔄 Initializing Git repository...")
        run_command(["git", "init"])
        print("✅ Git repository initialized.")
    
    # Add files
    print("🔄 Adding files to Git...")
    run_command(["git", "add", "."])
    
    # Create initial commit
    commit_result = run_command(["git", "status", "--porcelain"])
    if commit_result:  # There are changes to commit
        run_command(["git", "commit", "-m", "Initial project setup"])
        print("✅ Initial commit created.")
    
    # Create and switch to abiruth branch
    current_branch = run_command(["git", "branch", "--show-current"])
    if current_branch != "abiruth":
        print("🔄 Creating and switching to 'abiruth' branch...")
        run_command(["git", "checkout", "-b", "abiruth"])
        print("✅ Switched to 'abiruth' branch.")
    else:
        print("✅ Already on 'abiruth' branch.")
    
    return True


def check_optional_dependencies():
    """Check for optional system dependencies."""
    print_step(7, "Checking optional dependencies...")
    
    optional_deps = {
        "Neo4j": "neo4j",
        "Redis": "redis-server", 
        "Elasticsearch": "elasticsearch"
    }
    
    available = []
    missing = []
    
    for name, command in optional_deps.items():
        if run_command(["which", command]) or run_command(["where", command]):
            available.append(name)
            print(f"✅ {name} found.")
        else:
            missing.append(name)
            print(f"❌ {name} not found.")
    
    if missing:
        print(f"\n📝 Optional dependencies not found: {', '.join(missing)}")
        print("   These are not required for basic functionality but enable advanced features.")
        print("   Installation instructions:")
        print("   - Neo4j: https://neo4j.com/download/")
        print("   - Redis: https://redis.io/download")
        print("   - Elasticsearch: https://www.elastic.co/downloads/elasticsearch")


def print_next_steps():
    """Print instructions for next steps."""
    print_header("Setup Complete! Next Steps:")
    
    steps = [
        "1. Activate the virtual environment:",
        "   Windows: venv\\Scripts\\activate",
        "   macOS/Linux: source venv/bin/activate",
        "",
        "2. Edit configuration file:",
        "   config/.env - Add your API keys and database settings",
        "",
        "3. Test the installation:",
        "   python -m pytest tests/ -v",
        "",
        "4. Start data collection:",
        "   python -m data_acquisition.api_connectors --mp-api-key YOUR_KEY",
        "",
        "5. Run the development server:",
        "   python -m retrieval_embedding.api_server",
        "",
        "6. Explore the modules:",
        "   - data_acquisition/     # Data fetching and preprocessing",
        "   - kg_schema_fusion/     # Knowledge graph construction", 
        "   - retrieval_embedding/  # Vector search and embeddings",
        "   - rag_evaluation/       # RAG pipeline and evaluation",
        "",
        "7. Read the documentation:",
        "   README.md - Project overview and detailed instructions",
        "",
        "8. Join the development:",
        "   git checkout abiruth    # Switch to development branch",
        "   git checkout -b feature/your-feature  # Create feature branch"
    ]
    
    for step in steps:
        print(f"   {step}")
    
    print(f"\n{'='*60}")
    print(" Happy coding! 🚀")
    print(f"{'='*60}")


def main():
    """Main initialization function."""
    print_header("Accelerating Materials Discovery RAG - Project Setup")
    
    print("This script will set up your development environment for the")
    print("Accelerating Materials Discovery RAG project.")
    
    # Run setup steps
    success = True
    
    success &= check_python_version()
    success &= create_virtual_environment()
    
    if success:
        create_directories()
        setup_configuration()
        success &= setup_git()
        check_optional_dependencies()
    
    if success:
        print_header("Setup completed successfully!")
        
        # Ask about dependency installation
        while True:
            install_deps = input("\nWould you like to install Python dependencies now? (y/n): ").lower()
            if install_deps in ['y', 'yes']:
                install_dependencies()
                break
            elif install_deps in ['n', 'no']:
                print("📝 You can install dependencies later with:")
                if platform.system() == "Windows":
                    print("   venv\\Scripts\\pip install -r requirements.txt")
                else:
                    print("   venv/bin/pip install -r requirements.txt")
                break
            else:
                print("Please enter 'y' or 'n'.")
        
        print_next_steps()
    else:
        print_header("Setup incomplete!")
        print("❌ Some setup steps failed. Please check the errors above and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
