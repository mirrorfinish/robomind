"""
RoboMind AI Service Analyzer - Detect AI/ML inference services in ROS2 projects.

Scans Python source files for patterns indicating AI/ML service usage:
- vLLM / OpenAI-compatible API clients
- Ollama API clients
- YOLO / Ultralytics model loading
- Whisper speech recognition
- Piper TTS
- Triton Inference Server clients
- Generic HTTP AI service calls

This enables RoboMind to:
1. Map which nodes depend on which AI services
2. Identify GPU-heavy components
3. Understand the inference topology of a distributed robotics system
"""

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class AIServiceInfo:
    """Information about a detected AI/ML service."""
    name: str
    framework: str  # vllm, ollama, triton, ultralytics, whisper, piper, custom
    model_name: Optional[str] = None
    port: Optional[int] = None
    endpoint_path: Optional[str] = None
    host: Optional[str] = None
    file_path: Optional[Path] = None
    line_number: Optional[int] = None
    gpu_required: bool = True
    caller_files: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "framework": self.framework,
            "model_name": self.model_name,
            "port": self.port,
            "endpoint_path": self.endpoint_path,
            "host": self.host,
            "file_path": str(self.file_path) if self.file_path else None,
            "line_number": self.line_number,
            "gpu_required": self.gpu_required,
            "caller_files": self.caller_files,
        }


@dataclass
class AIServiceAnalysisResult:
    """Result of AI service analysis."""
    services: List[AIServiceInfo] = field(default_factory=list)
    files_scanned: int = 0
    files_with_ai: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "services": [s.to_dict() for s in self.services],
            "files_scanned": self.files_scanned,
            "files_with_ai": self.files_with_ai,
            "summary": {
                "total_services": len(self.services),
                "frameworks": list({s.framework for s in self.services}),
                "gpu_services": sum(1 for s in self.services if s.gpu_required),
            },
        }


# Regex patterns for detecting AI service usage
OPENAI_CLIENT_PATTERN = re.compile(
    r'OpenAI\s*\(\s*(?:.*?base_url\s*=\s*["\'](?P<url>[^"\']+)["\'])?'
)
OLLAMA_URL_PATTERN = re.compile(
    r'["\'](?P<url>https?://[^"\']*:11434[^"\']*)["\']'
)
VLLM_URL_PATTERN = re.compile(
    r'["\'](?P<url>https?://[^"\']*:(?:30000|30001|8000)[^"\']*(?:/v1)?)["\']'
)
YOLO_PATTERN = re.compile(
    r'(?:YOLO|YOLOv\d+)\s*\(\s*["\']?(?P<model>[^"\')\s]+)'
)
WHISPER_PATTERN = re.compile(
    r'whisper\.load_model\s*\(\s*["\'](?P<model>[^"\']+)["\']'
)
FASTER_WHISPER_PATTERN = re.compile(
    r'WhisperModel\s*\(\s*["\'](?P<model>[^"\']+)["\']'
)
PIPER_PATTERN = re.compile(
    r'(?:PiperVoice|piper_tts).*?["\'](?P<model>[^"\']+\.onnx)["\']'
)
TRITON_PATTERN = re.compile(
    r'(?:grpcclient|httpclient)\.InferenceServerClient\s*\(\s*["\'](?P<url>[^"\']+)["\']'
)

# Import patterns to detect frameworks
IMPORT_PATTERNS = {
    "openai": re.compile(r'(?:from\s+openai|import\s+openai)'),
    "ultralytics": re.compile(r'(?:from\s+ultralytics|import\s+ultralytics)'),
    "whisper": re.compile(r'(?:from\s+(?:faster_)?whisper|import\s+(?:faster_)?whisper)'),
    "ollama": re.compile(r'(?:from\s+ollama|import\s+ollama)'),
    "tritonclient": re.compile(r'(?:from\s+tritonclient|import\s+tritonclient)'),
    "transformers": re.compile(r'(?:from\s+transformers|import\s+transformers)'),
    "piper": re.compile(r'(?:from\s+piper|import\s+piper)'),
}


class AIServiceAnalyzer:
    """
    Analyze Python files for AI/ML service patterns.

    Usage:
        analyzer = AIServiceAnalyzer()
        result = analyzer.analyze_files(python_files)
        for service in result.services:
            print(f"{service.name}: {service.framework} on port {service.port}")
    """

    def __init__(self):
        self._seen_services: Dict[str, AIServiceInfo] = {}

    def analyze_files(self, file_paths: List[Path]) -> AIServiceAnalysisResult:
        """Analyze a list of Python files for AI service patterns."""
        result = AIServiceAnalysisResult()
        files_with_ai = set()

        for file_path in file_paths:
            result.files_scanned += 1
            services = self.analyze_file(file_path)
            if services:
                files_with_ai.add(str(file_path))

        result.files_with_ai = len(files_with_ai)
        result.services = list(self._seen_services.values())
        return result

    def analyze_file(self, file_path: Path) -> List[AIServiceInfo]:
        """Analyze a single Python file for AI service patterns."""
        try:
            content = file_path.read_text(errors="replace")
        except Exception as e:
            logger.debug(f"Could not read {file_path}: {e}")
            return []

        services = []

        # Detect frameworks from imports
        detected_imports = set()
        for framework, pattern in IMPORT_PATTERNS.items():
            if pattern.search(content):
                detected_imports.add(framework)

        # Detect OpenAI-compatible clients (vLLM, Ollama with OpenAI compat)
        for match in OPENAI_CLIENT_PATTERN.finditer(content):
            url = match.group("url")
            line_num = content[:match.start()].count("\n") + 1

            if url:
                port = self._extract_port(url)
                host = self._extract_host(url)
                # Determine if vLLM or Ollama based on port
                if port == 11434:
                    svc = self._register_service(
                        name=f"ollama_{host or 'local'}",
                        framework="ollama",
                        port=port,
                        host=host,
                        endpoint_path="/v1/chat/completions",
                        file_path=file_path,
                        line_number=line_num,
                    )
                else:
                    svc = self._register_service(
                        name=f"vllm_{host or 'local'}_{port or 30000}",
                        framework="vllm",
                        port=port,
                        host=host,
                        endpoint_path="/v1/chat/completions",
                        file_path=file_path,
                        line_number=line_num,
                    )
                services.append(svc)

        # Detect Ollama URLs
        for match in OLLAMA_URL_PATTERN.finditer(content):
            url = match.group("url")
            line_num = content[:match.start()].count("\n") + 1
            host = self._extract_host(url)
            svc = self._register_service(
                name=f"ollama_{host or 'local'}",
                framework="ollama",
                port=11434,
                host=host,
                endpoint_path=self._extract_path(url),
                file_path=file_path,
                line_number=line_num,
            )
            services.append(svc)

        # Detect vLLM URLs (port 30000/30001)
        for match in VLLM_URL_PATTERN.finditer(content):
            url = match.group("url")
            line_num = content[:match.start()].count("\n") + 1
            port = self._extract_port(url)
            host = self._extract_host(url)
            # Skip if already detected via OpenAI client
            key = f"vllm_{host or 'local'}_{port or 30000}"
            if key not in self._seen_services:
                svc = self._register_service(
                    name=key,
                    framework="vllm",
                    port=port,
                    host=host,
                    endpoint_path=self._extract_path(url),
                    file_path=file_path,
                    line_number=line_num,
                )
                services.append(svc)

        # Detect YOLO
        for match in YOLO_PATTERN.finditer(content):
            model = match.group("model")
            line_num = content[:match.start()].count("\n") + 1
            svc = self._register_service(
                name=f"yolo_{model}",
                framework="ultralytics",
                model_name=model,
                file_path=file_path,
                line_number=line_num,
            )
            services.append(svc)

        # Detect Whisper
        for match in WHISPER_PATTERN.finditer(content):
            model = match.group("model")
            line_num = content[:match.start()].count("\n") + 1
            svc = self._register_service(
                name=f"whisper_{model}",
                framework="whisper",
                model_name=model,
                file_path=file_path,
                line_number=line_num,
            )
            services.append(svc)

        # Detect faster-whisper
        for match in FASTER_WHISPER_PATTERN.finditer(content):
            model = match.group("model")
            line_num = content[:match.start()].count("\n") + 1
            svc = self._register_service(
                name=f"faster_whisper_{model}",
                framework="whisper",
                model_name=model,
                file_path=file_path,
                line_number=line_num,
            )
            services.append(svc)

        # Detect Piper TTS
        for match in PIPER_PATTERN.finditer(content):
            model = match.group("model")
            line_num = content[:match.start()].count("\n") + 1
            svc = self._register_service(
                name=f"piper_{Path(model).stem}",
                framework="piper",
                model_name=model,
                file_path=file_path,
                line_number=line_num,
                gpu_required=False,
            )
            services.append(svc)

        # Detect Triton
        for match in TRITON_PATTERN.finditer(content):
            url = match.group("url")
            line_num = content[:match.start()].count("\n") + 1
            host = self._extract_host(url)
            port = self._extract_port(url)
            svc = self._register_service(
                name=f"triton_{host or 'local'}",
                framework="triton",
                port=port,
                host=host,
                file_path=file_path,
                line_number=line_num,
            )
            services.append(svc)

        return services

    def _register_service(
        self,
        name: str,
        framework: str,
        port: Optional[int] = None,
        host: Optional[str] = None,
        model_name: Optional[str] = None,
        endpoint_path: Optional[str] = None,
        file_path: Optional[Path] = None,
        line_number: Optional[int] = None,
        gpu_required: bool = True,
    ) -> AIServiceInfo:
        """Register or update an AI service, deduplicating by name."""
        if name in self._seen_services:
            svc = self._seen_services[name]
            # Add caller file if not already tracked
            if file_path and str(file_path) not in svc.caller_files:
                svc.caller_files.append(str(file_path))
            # Update missing fields
            if model_name and not svc.model_name:
                svc.model_name = model_name
            if port and not svc.port:
                svc.port = port
            return svc

        svc = AIServiceInfo(
            name=name,
            framework=framework,
            port=port,
            host=host,
            model_name=model_name,
            endpoint_path=endpoint_path,
            file_path=file_path,
            line_number=line_number,
            gpu_required=gpu_required,
            caller_files=[str(file_path)] if file_path else [],
        )
        self._seen_services[name] = svc
        return svc

    @staticmethod
    def _extract_port(url: str) -> Optional[int]:
        """Extract port number from a URL string."""
        match = re.search(r':(\d{2,5})(?:/|$|\?)', url)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def _extract_host(url: str) -> Optional[str]:
        """Extract hostname from a URL string."""
        match = re.search(r'//([^:/]+)', url)
        if match:
            host = match.group(1)
            if host not in ("localhost", "127.0.0.1", "0.0.0.0"):
                return host
        return None

    @staticmethod
    def _extract_path(url: str) -> Optional[str]:
        """Extract path from a URL string."""
        match = re.search(r':\d+(/[^\s"\']*)', url)
        if match:
            return match.group(1)
        return None


def analyze_ai_services(file_paths: List[Path]) -> AIServiceAnalysisResult:
    """Convenience function to analyze files for AI services."""
    analyzer = AIServiceAnalyzer()
    return analyzer.analyze_files(file_paths)
