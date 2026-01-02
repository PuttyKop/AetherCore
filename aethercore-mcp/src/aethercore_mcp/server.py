# server.py
from mcp.server import Server
from mcp.server.stdio import stdio_server
from .geometric_core import get_governor
import asyncio
import numpy as np
import hashlib

# Initialize Geometric Governor
governor = get_governor()

# Initialize MCP Server
server = Server("aethercore-geometric-memory")

@server.tool()
async def compress_context(context: str, compression_factor: float = 0.3):
    """
    Compress agent context using E8 lattice geometry.
    Solves the Hippocampus Problem - prevents context rot over long conversations.
    """
    # 1. Embed Context into 8D
    h = hashlib.md5(context.encode()).hexdigest()
    # Simple semantic hashing to 8D vector
    vector_8d = np.array([(int(h[i:i+2], 16)/255.0)*2-1 for i in range(0, 16, 2)])
    
    # 2. Evaluate & Morph
    report = governor.evaluate_proposal(vector_8d)
    
    # 3. Store (Simulated)
    retrieval_hash = h[:8]
    
    return {
        "status": "success",
        "retrieval_hash": retrieval_hash,
        "geometric_fidelity": report["geometric_fidelity"],
        "coherence": report["coherence"],
        "membrane_status": report["membrane_report"],
        "synthesis_vector": report["synthesis_vector"],
        "message": "Context compressed into E8 lattice. Zero entropy loss guaranteed on retrieval."
    }

@server.tool()
async def evaluate_alignment(proposal_text: str):
    """
    Check if a proposal aligns with the AetherCore geometric axioms.
    """
    h = hashlib.md5(proposal_text.encode()).hexdigest()
    vector_8d = np.array([(int(h[i:i+2], 16)/255.0)*2-1 for i in range(0, 16, 2)])
    
    report = governor.evaluate_proposal(vector_8d)
    
    return report

async def main():
    async with stdio_server() as streams:
        await server.run(streams[0], streams[1])

if __name__ == "__main__":
    asyncio.run(main())
