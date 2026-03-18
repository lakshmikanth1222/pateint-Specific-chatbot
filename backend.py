from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import List, Optional, Dict, Any
import uvicorn
from urllib.parse import urlparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from dotenv import load_dotenv
import json
import asyncpg
from datetime import datetime
import uuid

# Load environment variables
load_dotenv()

# LlamaIndex Core & Database
from llama_index.core import Settings, PromptTemplate
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ==========================================
# 1. CONFIGURATION
# ==========================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
NEON_DATABASE_URI = os.getenv("NEON_DATABASE_URI", "")

print(f"Database URI: {NEON_DATABASE_URI[:50]}...")

# ==========================================
# 2. FASTAPI INITIALIZATION
# ==========================================
app = FastAPI(title="Medical AI Assistant API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a thread pool
executor = ThreadPoolExecutor()

# ==========================================
# 3. PYDANTIC MODELS (FIXED FOR UUID)
# ==========================================

class Patient(BaseModel):
    patient_id: str  # Will store UUID as string
    abha_id: str
    name: str
    date_of_birth: str
    gender: str
    phone_number: str
    created_at: Optional[str] = None

    @field_validator('patient_id', mode='before')
    @classmethod
    def convert_uuid_to_str(cls, v):
        """Convert UUID to string if necessary"""
        if isinstance(v, uuid.UUID):
            return str(v)
        return v

class ChatRequest(BaseModel):
    message: str
    patient_id: Optional[str] = None
    session_id: Optional[str] = None

class SourceNode(BaseModel):
    file_name: str
    patient_id: str
    abha_id: str
    score: Optional[float] = None
    text: str
    metadata: Dict[str, Any] = {}

    @field_validator('patient_id', mode='before')
    @classmethod
    def convert_uuid_to_str(cls, v):
        """Convert UUID to string if necessary"""
        if isinstance(v, uuid.UUID):
            return str(v)
        return v

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceNode]
    session_id: str
    patient_id: Optional[str] = None
    patient_info: Optional[Dict] = None

class HealthResponse(BaseModel):
    status: str
    gemini_api: str
    database: str
    patients_count: int
    message: str

class PatientsResponse(BaseModel):
    patients: List[Patient]
    total: int

# ==========================================
# 4. DATABASE CONNECTION
# ==========================================
class DatabaseManager:
    def __init__(self):
        self.pool = None
    
    async def initialize(self):
        """Initialize connection pool"""
        try:
            print("Initializing database connection...")
            self.pool = await asyncpg.create_pool(
                NEON_DATABASE_URI,
                min_size=1,
                max_size=10,
                command_timeout=60
            )
            print("[SUCCESS] Database pool created successfully")
            
            # Test connection
            async with self.pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                print(f"[SUCCESS] Database test query successful: {result}")
            
            return True
        except Exception as e:
            print(f"[ERROR] Database initialization error: {e}")
            return False
    
    async def get_patient_by_id(self, patient_id: str) -> Optional[Dict]:
        """Get patient details by ID"""
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT patient_id, abha_id, name, date_of_birth::text, gender, phone_number, created_at::text FROM patient WHERE patient_id = $1",
                    patient_id
                )
                if row:
                    # Convert UUID to string
                    result = dict(row)
                    if isinstance(result['patient_id'], uuid.UUID):
                        result['patient_id'] = str(result['patient_id'])
                    return result
                return None
        except Exception as e:
            print(f"Error getting patient by ID: {e}")
            return None
    
    async def get_all_patients(self) -> List[Dict]:
        """Get all patients"""
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT patient_id, abha_id, name, date_of_birth::text, gender, phone_number, created_at::text FROM patient ORDER BY name"
                )
                patients = []
                for row in rows:
                    patient = dict(row)
                    # Convert UUID to string
                    if isinstance(patient['patient_id'], uuid.UUID):
                        patient['patient_id'] = str(patient['patient_id'])
                    patients.append(patient)
                
                print(f"Found {len(patients)} patients in database")
                return patients
        except Exception as e:
            print(f"Error getting all patients: {e}")
            return []
    
    async def search_patient_records(self, query_embedding: List[float], patient_id: Optional[str] = None, limit: int = 5):
        """Search patient records using vector similarity"""
        try:
            async with self.pool.acquire() as conn:
                # Convert embedding to string for PostgreSQL
                embedding_str = '[' + ','.join(str(x) for x in query_embedding) + ']'
                
                if patient_id:
                    # Search with patient_id filter
                    rows = await conn.fetch("""
                        SELECT 
                            id,
                            text,
                            metadata,
                            1 - (embedding <=> $1::vector) as similarity_score
                        FROM patient_records
                        WHERE metadata->>'patient_id' = $2
                        ORDER BY embedding <=> $1::vector
                        LIMIT $3
                    """, embedding_str, patient_id, limit)
                else:
                    # Search across all patients
                    rows = await conn.fetch("""
                        SELECT 
                            id,
                            text,
                            metadata,
                            1 - (embedding <=> $1::vector) as similarity_score
                        FROM patient_records
                        ORDER BY embedding <=> $1::vector
                        LIMIT $2
                    """, embedding_str, limit)
                
                return rows
        except Exception as e:
            print(f"Error searching patient records: {e}")
            return []
    
    async def close(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            print("Database pool closed")

# ==========================================
# 5. MEDICAL AI ENGINE
# ==========================================
class MedicalAIEngine:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.llm = None
        self.embed_model = None
        self.initialized = False
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize AI components"""
        async with self._lock:
            if self.initialized:
                return True
            
            try:
                print("Initializing AI Engine...")
                
                # Configure Gemini LLM
                if not GOOGLE_API_KEY:
                    print("⚠️ WARNING: GOOGLE_API_KEY is not set")
                else:
                    self.llm = GoogleGenAI(
                        model="models/gemini-2.5-flash",
                        api_key=GOOGLE_API_KEY,
                        temperature=0.7,
                        max_tokens=4096
                    )
                    print("[SUCCESS] Gemini LLM initialized")
                
                # Configure HuggingFace Embeddings
                self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
                print("[SUCCESS] Embedding model initialized")
                
                # Set in Settings for LlamaIndex
                if self.llm:
                    Settings.llm = self.llm
                Settings.embed_model = self.embed_model
                
                self.initialized = True
                print("[SUCCESS] AI Engine fully initialized")
                return True
            except Exception as e:
                print(f"❌ AI Engine initialization error: {str(e)}")
                self.initialized = False
                raise e
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            executor,
            self.embed_model.get_text_embedding,
            text
        )
        return embedding
    
    async def search_patient_records(self, message: str, patient_id: Optional[str] = None):
        """Core vector search function"""
        # Generate embedding for the query
        query_embedding = await self.generate_embedding(message)
        
        # Search in database
        results = await self.db_manager.search_patient_records(
            query_embedding=query_embedding,
            patient_id=patient_id,
            limit=5
        )
        
        # Format results
        sources = []
        context_texts = []
        
        for row in results:
            metadata = row['metadata']
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}
            
            # Convert UUID in metadata if present
            patient_id_val = metadata.get('patient_id', 'Unknown')
            if isinstance(patient_id_val, uuid.UUID):
                patient_id_val = str(patient_id_val)
            
            source = {
                'file_name': metadata.get('file_name', 'Unknown'),
                'patient_id': patient_id_val,
                'abha_id': metadata.get('abha_id', 'Unknown'),
                'score': float(row['similarity_score']) if row['similarity_score'] else 0.0,
                'text': row['text'][:500],
                'metadata': metadata
            }
            
            sources.append(source)
            context_texts.append(row['text'])
        
        return {
            'sources': sources,
            'context': "\n\n---\n\n".join(context_texts)
        }
    
    async def generate_response(self, message: str, context: str, patient_info: Optional[Dict] = None):
        """Generate AI response using Gemini"""
        if not self.llm:
            return "⚠️ Gemini API not configured. Please set GOOGLE_API_KEY in .env file."
        
        # Create patient context if available
        patient_context = ""
        if patient_info:
            try:
                dob = datetime.strptime(patient_info.get('date_of_birth', ''), '%Y-%m-%d')
                age = datetime.now().year - dob.year
            except:
                age = "Unknown"
            
            patient_context = f"""
Patient Information:
- Name: {patient_info.get('name', 'Unknown')}
- Age: {age}
- Gender: {patient_info.get('gender', 'Unknown')}
- ABHA ID: {patient_info.get('abha_id', 'Unknown')}
"""
        
        # Medical system prompt
        system_prompt = f"""
You are an expert AI medical assistant. Analyze the patient's medical records and provide structured insights.

{patient_context}

### RULES:
1. Base your analysis STRICTLY on the provided medical records only
2. If information is not in records, say it's not available
3. Never provide definitive diagnoses - always suggest doctor review
4. Structure response with clear headers

### OUTPUT FORMAT:
- **Findings:**
- **Possible Conditions / Interpretation:**
- **Suggested Next Steps:**
- **Confidence Level:**

⚠️ *Disclaimer: This analysis is AI-generated based on uploaded records and must be reviewed by a qualified medical professional.*
"""
        
        full_prompt = f"{system_prompt}\n\nMedical Records:\n{context}\n\nQuestion: {message}\n\nAnswer:"
        
        # Generate response
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            executor,
            self.llm.complete,
            full_prompt
        )
        
        return str(response)

# ==========================================
# 6. INITIALIZE SERVICES
# ==========================================
db_manager = DatabaseManager()
ai_engine = MedicalAIEngine(db_manager)

# ==========================================
# 7. API ENDPOINTS
# ==========================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("\n" + "="*50)
    print("Starting Medical AI Assistant API...")
    print("="*50)
    
    # Initialize database
    db_success = await db_manager.initialize()
    if not db_success:
        print("[WARNING] Database initialization failed")
    
    # Initialize AI engine
    try:
        await ai_engine.initialize()
    except Exception as e:
        print(f"[WARNING] AI Engine initialization failed: {e}")
    
    # Test patient fetch
    if db_success:
        patients = await db_manager.get_all_patients()
        print(f"\n[STATUS] Database Status:")
        print(f"   - Total patients: {len(patients)}")
        if patients:
            print(f"   - Sample patient: {patients[0]['name']} (ID: {patients[0]['patient_id']})")
    
    print("="*50 + "\n")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        gemini_status = "Connected" if GOOGLE_API_KEY and GOOGLE_API_KEY != "" else "Missing"
        
        # Test database connection
        patients = []
        db_status = "Disconnected"
        patients_count = 0
        
        try:
            patients = await db_manager.get_all_patients()
            patients_count = len(patients)
            db_status = "Connected"
        except Exception as e:
            print(f"Health check database error: {e}")
            db_status = "Error"
        
        status = "healthy" if gemini_status == "Connected" and db_status == "Connected" else "degraded"
        
        return HealthResponse(
            status=status,
            gemini_api=gemini_status,
            database=db_status,
            patients_count=patients_count,
            message="Medical AI Assistant API is running"
        )
    except Exception as e:
        print(f"Health check error: {e}")
        return HealthResponse(
            status="unhealthy",
            gemini_api="Error",
            database="Error",
            patients_count=0,
            message=str(e)
        )

@app.get("/patients", response_model=PatientsResponse)
async def get_patients():
    """Get all patients"""
    try:
        print("Fetching all patients...")
        patients_data = await db_manager.get_all_patients()
        print(f"Found {len(patients_data)} patients")
        
        patients = []
        for p in patients_data:
            try:
                patient = Patient(**p)
                patients.append(patient)
                print(f"✅ Added patient: {patient.name}")
            except Exception as e:
                print(f"Error creating Patient model for {p.get('name')}: {e}")
        
        return PatientsResponse(patients=patients, total=len(patients))
    except Exception as e:
        print(f"Error in /patients endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/patients/{patient_id}", response_model=Patient)
async def get_patient(patient_id: str):
    """Get patient by ID"""
    try:
        patient_data = await db_manager.get_patient_by_id(patient_id)
        if not patient_data:
            raise HTTPException(status_code=404, detail="Patient not found")
        return Patient(**patient_data)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /patients/{patient_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint with vector search"""
    try:
        # Initialize if needed
        if not ai_engine.initialized:
            await ai_engine.initialize()
        
        # Get patient info if patient_id provided
        patient_info = None
        if request.patient_id:
            patient_info = await db_manager.get_patient_by_id(request.patient_id)
            if not patient_info:
                raise HTTPException(status_code=404, detail="Patient not found")
        
        print(f"🔍 Query: {request.message}")
        print(f"👤 Patient: {request.patient_id or 'All patients'}")
        
        # Step 1: Vector search
        search_results = await ai_engine.search_patient_records(
            message=request.message,
            patient_id=request.patient_id
        )
        
        print(f"📚 Found {len(search_results['sources'])} relevant sources")
        
        if not search_results['sources']:
            return ChatResponse(
                answer="No relevant medical records found for this query.",
                sources=[],
                session_id=request.session_id or "default",
                patient_id=request.patient_id,
                patient_info=patient_info
            )
        
        # Step 2: Generate AI response
        answer = await ai_engine.generate_response(
            message=request.message,
            context=search_results['context'],
            patient_info=patient_info
        )
        
        return ChatResponse(
            answer=answer,
            sources=[SourceNode(**source) for source in search_results['sources']],
            session_id=request.session_id or "default",
            patient_id=request.patient_id,
            patient_info=patient_info
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def vector_search(message: str, patient_id: Optional[str] = None):
    """Pure vector search endpoint"""
    try:
        if not ai_engine.initialized:
            await ai_engine.initialize()
        
        results = await ai_engine.search_patient_records(message, patient_id)
        return results
    except Exception as e:
        print(f"Error in /search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await db_manager.close()
    executor.shutdown(wait=True)
    print("Shutdown complete")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)