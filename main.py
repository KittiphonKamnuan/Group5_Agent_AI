# -*- coding: utf-8 -*-
"""
Thammasat AI Agent FastAPI Backend
สำหรับ expose AI Agent เป็น REST API endpoints
"""

import os
import logging
import traceback
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio
import json

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import required libraries สำหรับ AI Agent
import google.generativeai as genai
from tavily import TavilyClient
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Thammasat AI Agent API",
    description="API สำหรับ AI Agent ของมหาวิทยาลัยธรรมศาสตร์",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware configuration สำหรับ Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Next.js default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== PYDANTIC MODELS สำหรับ API REQUESTS/RESPONSES =====

class ChatRequest(BaseModel):
    """Model สำหรับ chat request"""
    message: str = Field(..., description="ข้อความจากผู้ใช้")
    session_id: Optional[str] = Field(None, description="Session ID สำหรับเก็บ context")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "มหาวิทยาลัยธรรมศาสตร์มีกี่วิทยาเขต",
                "session_id": "user123"
            }
        }

class ChatResponse(BaseModel):
    """Model สำหรับ chat response"""
    success: bool = Field(..., description="สถานะการประมวลผล")
    message: str = Field(..., description="คำตอบจาก AI")
    session_id: str = Field(..., description="Session ID")
    response_time: float = Field(..., description="เวลาที่ใช้ในการตอบ (วินาที)")
    sources_used: List[str] = Field(default=[], description="แหล่งข้อมูลที่ใช้")

class HealthResponse(BaseModel):
    """Model สำหรับ health check response"""
    status: str = Field(..., description="สถานะของ API")
    timestamp: datetime = Field(..., description="เวลาปัจจุบัน")
    version: str = Field(..., description="เวอร์ชันของ API")
    services: Dict[str, bool] = Field(..., description="สถานะของ services ต่างๆ")

class ErrorResponse(BaseModel):
    """Model สำหรับ error response"""
    success: bool = Field(False, description="สถานะการประมวลผล")
    error: str = Field(..., description="ข้อความ error")
    error_code: str = Field(..., description="รหัส error")
    timestamp: datetime = Field(..., description="เวลาที่เกิด error")

# ===== CONFIGURATION CLASS =====

class APIConfig:
    """Class สำหรับจัดการ configuration ของ API"""
    
    def __init__(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.max_message_length = int(os.getenv("MAX_MESSAGE_LENGTH", "1000"))
        self.response_timeout = int(os.getenv("RESPONSE_TIMEOUT", "30"))
        self.enable_pdf_upload = os.getenv("ENABLE_PDF_UPLOAD", "true").lower() == "true"
        
        # Validate required API keys
        self._validate_config()
        
    def _validate_config(self):
        """ตรวจสอบ configuration ที่จำเป็น"""
        if not self.google_api_key:
            logger.error("GOOGLE_API_KEY is not set")
            raise ValueError("GOOGLE_API_KEY environment variable is required")
            
        if not self.tavily_api_key:
            logger.warning("TAVILY_API_KEY is not set - web search will be disabled")
            
    def get_service_status(self) -> Dict[str, bool]:
        """ตรวจสอบสถานะของ services ต่างๆ"""
        return {
            "gemini_api": bool(self.google_api_key),
            "tavily_search": bool(self.tavily_api_key),
            "pdf_upload": self.enable_pdf_upload
        }

# ===== AI AGENT CLASS =====

class ThammasatAIAgent:
    """
    Simplified version ของ AI Agent สำหรับ production
    ได้รับการปรับปรุงให้เหมาะสมกับการใช้งานแบบ API
    """
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.sessions: Dict[str, List[Dict]] = {}  # เก็บ conversation history แต่ละ session
        
        # Initialize Gemini
        if config.google_api_key:
            genai.configure(api_key=config.google_api_key)
            self.model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                generation_config={
                    "temperature": 0.1,
                    "top_p": 0.95,
                    "top_k": 64,
                    "max_output_tokens": 3000,
                }
            )
            logger.info("Gemini model initialized successfully")
        else:
            self.model = None
            logger.error("Cannot initialize Gemini - API key missing")
            
        # Initialize Tavily search (optional)
        self.tavily_client = None
        if config.tavily_api_key:
            try:
                self.tavily_client = TavilyClient(api_key=config.tavily_api_key)
                logger.info("Tavily search client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Tavily: {e}")
                
        # Load Thammasat sites data
        self.thammasat_sites = self._load_thammasat_sites()
        
    def _load_thammasat_sites(self) -> List[Dict[str, str]]:
        """โหลดข้อมูลเว็บไซต์ของมหาวิทยาลัยธรรมศาสตร์"""
        sites_data = [
            {"domain": "มหาวิทยาลัยหลัก", "site": "tu.ac.th", "description": "เว็บไซต์หลักมหาวิทยาลัยธรรมศาสตร์"},
            {"domain": "การรับสมัคร", "site": "tuadmissions.in.th", "description": "ระบบรับสมัครนักศึกษา"},
            {"domain": "การลงทะเบียน", "site": "reg.tu.ac.th", "description": "ระบบลงทะเบียนเรียน"},
            {"domain": "หอสมุด", "site": "library.tu.ac.th", "description": "หอสมุดมหาวิทยาลัยธรรมศาสตร์"},
            {"domain": "คณะนิติศาสตร์", "site": "law.tu.ac.th", "description": "คณะนิติศาสตร์"},
            {"domain": "คณะพาณิชยศาสตร์", "site": "tbs.tu.ac.th", "description": "คณะพาณิชยศาสตรและการบัญชี"},
            {"domain": "คณะรัฐศาสตร์", "site": "polsci.tu.ac.th", "description": "คณะรัฐศาสตร์"},
            {"domain": "คณะเศรษฐศาสตร์", "site": "econ.tu.ac.th", "description": "คณะเศรษฐศาสตร์"},
        ]
        return sites_data
        
    def _get_relevant_sites(self, query: str) -> List[str]:
        """หาเว็บไซต์ที่เกี่ยวข้องกับคำถาม"""
        query_lower = query.lower()
        relevant_sites = []
        
        # Keyword mapping
        keyword_mapping = {
            'รับสมัคร': ['tuadmissions.in.th'],
            'ลงทะเบียน': ['reg.tu.ac.th'],
            'หอสมุด': ['library.tu.ac.th'],
            'นิติศาสตร์': ['law.tu.ac.th'],
            'พาณิชยศาสตร์': ['tbs.tu.ac.th'],
            'รัฐศาสตร์': ['polsci.tu.ac.th'],
            'เศรษฐศาสตร์': ['econ.tu.ac.th'],
        }
        
        for keyword, sites in keyword_mapping.items():
            if keyword in query_lower:
                relevant_sites.extend(sites)
                
        # Default sites if no specific match
        if not relevant_sites:
            relevant_sites = ['tu.ac.th']
            
        return list(set(relevant_sites))[:3]  # จำกัดไม่เกิน 3 sites
        
    async def search_web(self, query: str, sites: List[str]) -> str:
        """ค้นหาข้อมูลจากเว็บไซต์"""
        if not self.tavily_client:
            return "การค้นหาเว็บไม่พร้อมใช้งาน - Tavily API key ไม่ได้ตั้งค่า"
            
        try:
            search_results = self.tavily_client.search(
                query=query,
                max_results=5,
                search_depth="basic",
                include_domains=sites,
            )
            
            if not search_results.get("results"):
                return "ไม่พบผลลัพธ์จากการค้นหาเว็บไซต์"
                
            # Format results
            formatted_results = []
            for i, result in enumerate(search_results["results"], 1):
                title = result.get("title", "ไม่มีหัวข้อ")
                url = result.get("url", "")
                content = result.get("content", "")[:1000]  # จำกัดความยาว
                
                formatted_results.append(f"""
ผลลัพธ์ที่ {i}:
หัวข้อ: {title}
URL: {url}
เนื้อหา: {content}
---""")
                
            return "\n".join(formatted_results)
            
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return f"เกิดข้อผิดพลาดในการค้นหา: {str(e)}"
            
    def _get_session_context(self, session_id: str) -> str:
        """ดึง context จากการสนทนาก่อนหน้า"""
        if session_id not in self.sessions:
            return ""
            
        history = self.sessions[session_id][-4:]  # เอาแค่ 4 ข้อความล่าสุด
        context_parts = []
        
        for entry in history:
            context_parts.append(f"{entry['role']}: {entry['content']}")
            
        return "\n".join(context_parts) if context_parts else ""
        
    def _update_session(self, session_id: str, user_message: str, ai_response: str):
        """อัพเดท conversation history"""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
            
        self.sessions[session_id].extend([
            {"role": "ผู้ใช้", "content": user_message, "timestamp": datetime.now()},
            {"role": "ผู้ช่วย", "content": ai_response, "timestamp": datetime.now()}
        ])
        
        # จำกัดประวัติไม่เกิน 20 ข้อความ
        if len(self.sessions[session_id]) > 20:
            self.sessions[session_id] = self.sessions[session_id][-20:]
            
    async def chat(self, message: str, session_id: str = "default") -> Dict[str, Any]:
        """
        ประมวลผลข้อความจากผู้ใช้และส่งคืนคำตอบ
        
        Args:
            message: ข้อความจากผู้ใช้
            session_id: ID ของ session สำหรับเก็บ context
            
        Returns:
            Dict ที่มีคำตอบและข้อมูลเพิ่มเติม
        """
        start_time = asyncio.get_event_loop().time()
        sources_used = []
        
        try:
            if not self.model:
                raise ValueError("Gemini model ไม่พร้อมใช้งาน")
                
            # 1. หาเว็บไซต์ที่เกี่ยวข้อง
            relevant_sites = self._get_relevant_sites(message)
            sources_used.extend(relevant_sites)
            
            # 2. ค้นหาข้อมูลจากเว็บ
            web_results = ""
            if self.tavily_client and relevant_sites:
                web_results = await self.search_web(message, relevant_sites)
                
            # 3. ดึง context จากการสนทนาก่อนหน้า
            conversation_context = self._get_session_context(session_id)
            
            # 4. สร้าง prompt สำหรับ Gemini
            system_prompt = f"""คุณเป็นผู้ช่วยตอบคำถามเกี่ยวกับมหาวิทยาลัยธรรมศาสตร์

แหล่งข้อมูลที่มีอยู่:
{json.dumps(self.thammasat_sites, ensure_ascii=False, indent=2)}

ผลลัพธ์การค้นหาจากเว็บไซต์:
{web_results}

บริบทการสนทนาก่อนหน้า:
{conversation_context}

คำแนะนำ:
1. ตอบคำถามตามข้อมูลที่ได้จากการค้นหา
2. ให้คำตอบที่ถูกต้องและครอบคลุม
3. หากไม่มีข้อมูลเพียงพอ ให้บอกอย่างชัดเจน
4. ตอบเป็นภาษาไทยที่สุภาพและเป็นมิตร
5. ระบุแหล่งที่มาหากมีข้อมูลจากเว็บไซต์

คำถาม: {message}

คำตอบ:"""

            # 5. ส่งไปให้ Gemini ประมวลผล
            response = await asyncio.to_thread(
                self.model.generate_content,
                system_prompt
            )
            
            if not response or not response.text:
                raise ValueError("ไม่สามารถสร้างคำตอบได้")
                
            ai_response = response.text.strip()
            
            # 6. อัพเดท session history
            self._update_session(session_id, message, ai_response)
            
            # คำนวณเวลาที่ใช้
            end_time = asyncio.get_event_loop().time()
            response_time = end_time - start_time
            
            return {
                "success": True,
                "message": ai_response,
                "session_id": session_id,
                "response_time": response_time,
                "sources_used": sources_used
            }
            
        except Exception as e:
            logger.error(f"Chat processing error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            end_time = asyncio.get_event_loop().time()
            response_time = end_time - start_time
            
            return {
                "success": False,
                "message": f"ขออภัย เกิดข้อผิดพลาดในการประมวลผล: {str(e)}",
                "session_id": session_id,
                "response_time": response_time,
                "sources_used": sources_used
            }

# ===== GLOBAL INSTANCES =====

# Initialize configuration และ AI agent
try:
    config = APIConfig()
    ai_agent = ThammasatAIAgent(config)
    logger.info("AI Agent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize AI Agent: {e}")
    config = None
    ai_agent = None

# ===== DEPENDENCY FUNCTIONS =====

def get_ai_agent() -> ThammasatAIAgent:
    """Dependency function สำหรับ get AI agent instance"""
    if ai_agent is None:
        raise HTTPException(
            status_code=503,
            detail="AI Agent is not available - check configuration"
        )
    return ai_agent

def get_config() -> APIConfig:
    """Dependency function สำหรับ get configuration"""
    if config is None:
        raise HTTPException(
            status_code=503,
            detail="Configuration is not available"
        )
    return config

# ===== API ENDPOINTS =====

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint สำหรับทดสอบ API"""
    return {
        "message": "Thammasat AI Agent API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check(config: APIConfig = Depends(get_config)):
    """Health check endpoint สำหรับตรวจสอบสถานะ API"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        services=config.get_service_status()
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    agent: ThammasatAIAgent = Depends(get_ai_agent)
):
    """
    Main chat endpoint สำหรับสนทนากับ AI Agent
    
    Args:
        request: ChatRequest object ที่มี message และ session_id
        agent: ThammasatAIAgent instance
        
    Returns:
        ChatResponse object ที่มีคำตอบและข้อมูลเพิ่มเติม
    """
    try:
        # Validate input
        if not request.message or len(request.message.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Message cannot be empty"
            )
            
        # ตั้ง session_id หากไม่ได้ระบุ
        session_id = request.session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # ประมวลผลข้อความ
        result = await agent.chat(request.message, session_id)
        
        if result["success"]:
            return ChatResponse(
                success=True,
                message=result["message"],
                session_id=result["session_id"],
                response_time=result["response_time"],
                sources_used=result["sources_used"]
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=result["message"]
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.delete("/chat/session/{session_id}")
async def clear_session(
    session_id: str,
    agent: ThammasatAIAgent = Depends(get_ai_agent)
):
    """ลบ conversation history ของ session"""
    try:
        if session_id in agent.sessions:
            del agent.sessions[session_id]
            return {"success": True, "message": f"Session {session_id} cleared"}
        else:
            return {"success": False, "message": f"Session {session_id} not found"}
    except Exception as e:
        logger.error(f"Clear session error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear session: {str(e)}"
        )

@app.get("/chat/sessions")
async def list_sessions(agent: ThammasatAIAgent = Depends(get_ai_agent)):
    """แสดงรายชื่อ active sessions"""
    try:
        sessions_info = {}
        for session_id, history in agent.sessions.items():
            sessions_info[session_id] = {
                "message_count": len(history),
                "last_activity": history[-1]["timestamp"].isoformat() if history else None
            }
        return {"sessions": sessions_info}
    except Exception as e:
        logger.error(f"List sessions error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list sessions: {str(e)}"
        )

# ===== ERROR HANDLERS =====

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_code=f"HTTP_{exc.status_code}",
            timestamp=datetime.now()
        ).model_dump()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            error_code="INTERNAL_ERROR",
            timestamp=datetime.now()
        ).model_dump()
    )

# ===== STARTUP/SHUTDOWN EVENTS =====

@app.on_event("startup")
async def startup_event():
    """เหตุการณ์เมื่อ API เริ่มทำงาน"""
    logger.info("🚀 Thammasat AI Agent API starting up...")
    logger.info(f"📊 Configuration loaded successfully")
    logger.info(f"🤖 AI Agent status: {'Ready' if ai_agent else 'Not available'}")

@app.on_event("shutdown")
async def shutdown_event():
    """เหตุการณ์เมื่อ API หยุดทำงาน"""
    logger.info("🛑 Thammasat AI Agent API shutting down...")

# ===== MAIN FUNCTION สำหรับรัน SERVER =====

if __name__ == "__main__":
    # รัน server ด้วย uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # เปิด auto-reload สำหรับ development
        log_level="info"
    )