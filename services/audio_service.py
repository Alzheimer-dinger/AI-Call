import wave
import io
import os
import datetime
import struct
import logging
from typing import List, Optional
from google.cloud import storage
from settings import SEND_SAMPLE_RATE

logger = logging.getLogger(__name__)

class StreamingAudioRecorder:
    """스트리밍 방식 오디오 녹음 및 GCS 업로드"""
    
    def __init__(self, user_id: str, session_id: str):
        self.user_id = user_id
        self.session_id = session_id
        self.gcs_client = storage.Client()
        self.bucket_name = os.getenv("GCS_BUCKET_NAME", "voice-recordings")
        
        # PCM 스트림 설정
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.pcm_blob_name = f"recordings/{user_id}/{session_id}_{timestamp}.pcm"
        self.wav_blob_name = f"recordings/{user_id}/{session_id}_{timestamp}.wav"
        
        # 업로드 준비
        self.bucket = self.gcs_client.bucket(self.bucket_name)
        self.pcm_blob = self.bucket.blob(self.pcm_blob_name)
        
        # PCM 스트림 업로드 시작 (resumable upload)
        self.pcm_stream = self.pcm_blob.open("wb")
        self.total_frames = 0
        
    async def append_audio_chunk(self, audio_chunk: bytes) -> bool:
        """오디오 청크를 실시간으로 GCS에 업로드"""
        try:
            if self.pcm_stream:
                self.pcm_stream.write(audio_chunk)
                self.total_frames += len(audio_chunk) // 2  # 16bit = 2bytes per sample
                return True
        except Exception as e:
            logger.error(f"PCM 스트림 업로드 실패: {e}")
            return False
        return False
            
    def _create_wav_header(self, sample_rate: int, num_channels: int, bits_per_sample: int, data_size: int) -> bytes:
        """WAV 파일 헤더 생성"""
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        
        header = struct.pack('<4sL4s4sLHHLLHH4sL',
            b'RIFF',
            36 + data_size,  # 파일 크기 - 8
            b'WAVE',
            b'fmt ',
            16,  # fmt chunk size
            1,   # PCM format
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b'data',
            data_size
        )
        return header
    
    async def finalize_recording(self) -> Optional[str]:
        """세션 종료시 PCM을 WAV로 변환하여 최종 파일 생성"""
        try:
            # PCM 스트림 닫기
            if self.pcm_stream:
                try:
                    self.pcm_stream.close()
                    logger.info(f"PCM 스트림 닫기 완료. 총 프레임: {self.total_frames}")
                except Exception as e:
                    logger.error(f"PCM 스트림 닫기 중 오류: {e}")
                finally:
                    self.pcm_stream = None
            
            if self.total_frames == 0:
                logger.warning("녹음된 오디오가 없습니다. (total_frames = 0)")
                return None
            
            # PCM 파일 크기 확인
            try:
                self.pcm_blob.reload()
                data_size = self.pcm_blob.size if self.pcm_blob.size else 0
                logger.info(f"PCM 파일 크기: {data_size} bytes")
            except Exception as e:
                logger.error(f"PCM 파일 정보 확인 중 오류: {e}")
                return None
            
            if data_size == 0:
                logger.warning("PCM 파일이 비어있습니다.")
                return None
            
            # WAV 헤더 생성 및 별도 블롭에 업로드
            wav_header = self._create_wav_header(
                sample_rate=SEND_SAMPLE_RATE,
                num_channels=1,
                bits_per_sample=16,
                data_size=data_size
            )
            
            header_blob_name = f"{self.pcm_blob_name}.header"
            header_blob = self.bucket.blob(header_blob_name)
            header_blob.upload_from_string(wav_header)
            
            # GCS Compose를 사용하여 헤더 + PCM 데이터 결합
            wav_blob = self.bucket.blob(self.wav_blob_name)
            wav_blob.compose([header_blob, self.pcm_blob])
            
            # 임시 파일들 삭제
            header_blob.delete()
            self.pcm_blob.delete()
            
            wav_url = f"gs://{self.bucket_name}/{self.wav_blob_name}"
            logger.info(f"WAV 파일 생성 완료: {wav_url}")
            return wav_url
            
        except Exception as e:
            logger.error(f"WAV 파일 생성 실패: {e}")
            return None
    
    def cleanup(self):
        """리소스 정리"""
        try:
            if self.pcm_stream:
                self.pcm_stream.close()
                self.pcm_stream = None
        except:
            pass

class AudioService:
    """음성 녹음 서비스 (레거시 지원)"""
    
    def __init__(self):
        # GCS 클라이언트 초기화
        self.gcs_client = storage.Client()
        self.bucket_name = os.getenv("GCS_BUCKET_NAME", "voice-recordings")
    
    def create_streaming_recorder(self, user_id: str, session_id: str) -> StreamingAudioRecorder:
        """스트리밍 녹음기 생성"""
        return StreamingAudioRecorder(user_id, session_id)
        
    def create_wav_file(self, audio_chunks: List[bytes], sample_rate: int = SEND_SAMPLE_RATE) -> bytes:
        """레거시: 오디오 청크들을 WAV 파일로 변환"""
        if not audio_chunks:
            return b""
            
        # WAV 파일을 메모리에 생성
        wav_buffer = io.BytesIO()
        
        # 모든 오디오 청크를 하나로 결합
        combined_audio = b''.join(audio_chunks)
        
        # WAV 파일 작성
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # 모노
            wav_file.setsampwidth(2)  # 16bit = 2 bytes
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(combined_audio)
        
        wav_buffer.seek(0)
        return wav_buffer.getvalue()
    
    def save_and_upload_recording(self, audio_chunks: List[bytes], user_id: str, session_id: str) -> Optional[str]:
        """레거시: 녹음을 WAV로 변환하고 GCS에 업로드 (메모리 방식)"""
        try:
            # WAV 파일 생성
            wav_data = self.create_wav_file(audio_chunks)
            
            if not wav_data:
                logger.warning("오디오 데이터가 비어있습니다.")
                return None
            
            # 파일명 생성
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            blob_name = f"recordings/{user_id}/{session_id}_{timestamp}.wav"
            
            # 버킷과 블롭 객체 생성
            bucket = self.gcs_client.bucket(self.bucket_name)
            blob = bucket.blob(blob_name)
            
            # 파일 업로드
            blob.upload_from_string(
                wav_data,
                content_type='audio/wav'
            )
            
            gcs_url = f"gs://{self.bucket_name}/{blob_name}"
            logger.info(f"녹음 파일 업로드 성공: {gcs_url}")
            return gcs_url
                
        except Exception as e:
            logger.error(f"녹음 저장 및 업로드 중 오류: {e}")
            return None

# 전역 오디오 서비스 인스턴스
audio_service = AudioService()