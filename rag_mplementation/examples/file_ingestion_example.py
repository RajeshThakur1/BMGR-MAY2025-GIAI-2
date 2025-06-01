"""Example of ingesting various file types."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.rag_pipeline import RAGPipeline

def create_sample_files():
    """Create sample files for testing."""
    sample_dir = Path("sample_documents")
    sample_dir.mkdir(exist_ok=True)
    
    # Create sample text file
    with open(sample_dir / "sample.txt", "w") as f:
        f.write("""
        This is a sample text document about artificial intelligence.
        AI has been transforming various industries including healthcare, finance, and transportation.
        Machine learning algorithms can learn patterns from data and make predictions.
        """)
    
    # Create sample markdown file
    with open(sample_dir / "sample.md", "w") as f:
        f.write("""
        # RAG Application Guide
        
        ## What is RAG?
        Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval 
        with text generation to provide more accurate and contextual responses.
        
        ## Benefits of RAG
        - Improved accuracy
        - Up-to-date information
        - Source attribution
        - Reduced hallucinations
        """)
    
    return sample_dir

def main():
    """File ingestion example."""
    print("📁 RAG Application - File Ingestion Example")
    print("=" * 50)
    
    # Create sample files
    sample_dir = create_sample_files()
    print(f"Created sample files in: {sample_dir}")
    
    # Initialize RAG pipeline
    rag = RAGPipeline()
    
    # Example 1: Ingest individual files
    print("\n📄 Example 1: Ingesting individual files")
    file_paths = [
        str(sample_dir / "sample.txt"),
        str(sample_dir / "sample.md")
    ]
    
    ingestion_stats = rag.ingest_documents(file_paths)
    print(f"Ingestion stats: {ingestion_stats}")
    
    # Example 2: Ingest entire directory
    print("\n📁 Example 2: Ingesting entire directory")
    directory_stats = rag.ingest_directory(str(sample_dir))
    print(f"Directory ingestion stats: {directory_stats}")
    
    # Example 3: Query ingested documents
    print("\n🔍 Example 3: Querying ingested documents")
    questions = [
        "What is artificial intelligence?",
        "What are the benefits of RAG?",
        "Tell me about machine learning"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        result = rag.query(question)
        print(f"A: {result['answer']}")
        
        # Show sources
        if result['source_documents']:
            print("Sources:")
            for doc in result['source_documents']:
                source = doc['metadata'].get('source', 'Unknown')
                print(f"  - {Path(source).name}")
    
    print("\n✅ File ingestion example completed!")
    print(f"Sample files are in: {sample_dir}")

if __name__ == "__main__":
    main() 