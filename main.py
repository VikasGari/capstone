from config.config_manager import ConfigManager
from src.ingestion.pipeline import IngestionPipeline
from src.interface.api import start_server

def main():
    # Load configuration
    config_manager = ConfigManager()
    
    # 1. Run Ingestion Pipeline (ensure database index is built and up to date)
    print("Running document ingestion pipeline...")
    pipeline = IngestionPipeline(config_manager)
    pipeline.run()
    print("Ingestion pipeline completed.")
    
    # 2. Start FastAPI Server
    print("Launching FastAPI backend server...")
    start_server()

if __name__ == "__main__":
    main()
