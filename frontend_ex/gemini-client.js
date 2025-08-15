// --- IMPORTANT : 1. Gemini API 통신 클래스 (⭐) ---
class GeminiAPI {
    constructor(endpoint, token = null) {
        this.endpoint = endpoint;
        this.token = token;
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000; // 1초
        this.isManualDisconnect = false;

        // 이벤트 핸들러 콜백
        this.onOpen = () => {}; // 서버와 연결
        this.onClose = () => {}; // 서버와 연결 해제
        this.onError = () => {}; // 서버 에러 발생 시
        this.onAudio = () => {}; // 오디오 청크 수신 시
        this.onInputTranscript = () => {}; // 발화 텍스트 수신 시 (청크 단위)
        this.onOutputTranscript = () => {}; // 응답 텍스트 수신 시 (청크 단위)
        this.onTurnComplete = () => {}; // 모델 응답 끝났을 시
        this.onInterrupt = () => {}; // 사용자가 중간에 말을 끊었을 시 
    }

    connect() {
        this.isManualDisconnect = false;
        // JWT 토큰을 URL 쿼리 매개변수로 추가
        const wsUrl = this.token ? `${this.endpoint}?token=${this.token}` : this.endpoint;
        this.ws = new WebSocket(wsUrl);
        this._setupWebSocketHandlers();
    }

    reconnect() {
        if (this.isManualDisconnect || this.reconnectAttempts >= this.maxReconnectAttempts) {
            return;
        }

        this.reconnectAttempts++;
        console.log(`재연결 시도 ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
        
        setTimeout(() => {
            this.connect();
        }, this.reconnectDelay * this.reconnectAttempts);
    }

    sendAudio(audioBuffer) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(audioBuffer);
        }
    }

    close() {
        this.isManualDisconnect = true;
        if (this.ws) {
            this.ws.close();
        }
    }

    _setupWebSocketHandlers() {
        this.ws.onopen = (event) => {
            console.log("WebSocket 연결 성공");
            this.reconnectAttempts = 0; // 연결 성공 시 재연결 카운트 리셋
            this.onOpen(event);
        };

        this.ws.onmessage = (event) => {
            try {
                const payload = JSON.parse(event.data);
                this._handleServerMessage(payload);
            } catch (error) {
                console.error("메시지 파싱 오류:", error);
            }
        };

        this.ws.onclose = (event) => {
            console.log("웹소켓 연결이 종료되었습니다.", event.reason);
            this.onClose(event);
            
            // 수동 종료가 아닌 경우에만 재연결 시도
            if (!this.isManualDisconnect && event.code !== 1000) {
                console.log("예상치 못한 연결 종료, 재연결 시도");
                this.reconnect();
            }
        };

        this.ws.onerror = (error) => {
            console.error("웹소켓 오류:", error);
            this.onError(error);
        };
    }

    _handleServerMessage(payload) {
        switch (payload.type) {
            case 'input_transcript':
                this.onInputTranscript(payload.data);
                break;
            case 'output_transcript':
                this.onOutputTranscript(payload.data);
                break;
            case 'audio':
                this.onAudio(payload.data);
                break;
            case 'turn_complete':
                this.onTurnComplete();
                break;
            case 'interrupt': // Interrupt 메시지 처리
                console.log("Interrupt 메시지 수신");
                this.onInterrupt();
                break;
            default:
                console.warn("알 수 없는 메시지 유형:", payload.type);
        }
    }
}

// --- 2. 오디오 스트리밍 재생 클래스 ---
class StreamingAudioPlayer {
    constructor(sampleRate) {
        this.sampleRate = sampleRate;
        this.audioContext = null;
        this.audioQueue = [];
        this.isPlaying = false;
        this.nextPlayTime = 0;
        this.activeSources = []; // 재생 중인 오디오 소스 추적
    }

    _ensureAudioContext() {
        if (!this.audioContext || this.audioContext.state === 'closed') {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: this.sampleRate });
            this.nextPlayTime = this.audioContext.currentTime;
        }
        if (this.audioContext.state === 'suspended') {
            this.audioContext.resume();
        }
    }

    receiveAudio(base64Audio) {
        this._ensureAudioContext();
        const binaryString = window.atob(base64Audio);
        const len = binaryString.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }

        const int16Array = new Int16Array(bytes.buffer);
        const float32Array = new Float32Array(int16Array.length);
        for (let i = 0; i < int16Array.length; i++) {
            float32Array[i] = int16Array[i] / 32768.0;
        }

        this.audioQueue.push(float32Array);

        if (!this.isPlaying) {
            this.isPlaying = true;
            this.scheduleNextChunk();
        }
    }

    scheduleNextChunk() {
        if (this.audioQueue.length === 0) {
            this.isPlaying = false;
            return;
        }
        // 오디오 컨텍스트 상태 복구 시도
        if(!this.audioContext || this.audioContext.state === 'closed') {
            console.log("오디오 컨텍스트 재생성 시도");
            this._ensureAudioContext();
            if(!this.audioContext || this.audioContext.state === 'closed') {
                console.error("오디오 컨텍스트 복구 실패");
                this.isPlaying = false;
                return;
            }
        }

        const audioChunk = this.audioQueue.shift();
        const buffer = this.audioContext.createBuffer(1, audioChunk.length, this.sampleRate);
        buffer.copyToChannel(audioChunk, 0);

        const source = this.audioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(this.audioContext.destination);

        const currentTime = this.audioContext.currentTime;
        const scheduleTime = this.nextPlayTime < currentTime ? currentTime : this.nextPlayTime;

        source.start(scheduleTime);
        this.nextPlayTime = scheduleTime + buffer.duration;

        // 활성 소스 목록에 추가하고, 재생이 끝나면 제거
        this.activeSources.push(source);
        source.onended = () => {
            this.activeSources = this.activeSources.filter(s => s !== source);
            if (this.isPlaying) {
                this.scheduleNextChunk();
            }
        };
        
        // 오디오 소스 에러 처리 추가
        source.onerror = (error) => {
            console.error("오디오 소스 에러:", error);
            this.activeSources = this.activeSources.filter(s => s !== source);
            if (this.isPlaying) {
                this.scheduleNextChunk();
            }
        };
    }

    // Interrupt를 처리하는 새로운 메소드
    interrupt() {
        console.log("오디오 재생 중단 및 버퍼 비우기");
        this.isPlaying = false;
        this.audioQueue = []; // 오디오 큐 비우기

        // 현재 재생 중이거나 스케줄된 모든 오디오 소스 중지
        this.activeSources.forEach(source => {
            try {
                source.stop(0);
            } catch (e) {
                // 이미 중지된 소스에 대해 stop()을 호출하면 발생하는 오류를 무시
            }
        });
        this.activeSources = []; // 활성 소스 목록 초기화

        // 다음 재생 시간 초기화
        if (this.audioContext) {
            this.nextPlayTime = this.audioContext.currentTime;
        }
    }

    stop() {
        this.interrupt(); // stop 호출 시 interrupt와 동일한 로직 수행
        if (this.audioContext && this.audioContext.state !== 'closed') {
            this.audioContext.close().catch(console.error);
            this.audioContext = null;
        }
    }
}

// --- 3. 마이크 입력 처리 클래스 ---
class Microphone {
    constructor(sampleRate, onAudioCallback) {
        this.sampleRate = sampleRate;
        this.onAudioCallback = onAudioCallback;
        this.mediaStream = null;
        this.audioContext = null;
        this.processor = null;
    }

    async start() {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: this.sampleRate
        });

        this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const source = this.audioContext.createMediaStreamSource(this.mediaStream);

        this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);
        this.processor.onaudioprocess = (e) => {
            const inputData = e.inputBuffer.getChannelData(0);
            const int16Array = new Int16Array(inputData.length);
            for (let i = 0; i < inputData.length; i++) {
                int16Array[i] = Math.max(-1, Math.min(1, inputData[i])) * 32767;
            }
            this.onAudioCallback(int16Array.buffer);
        };

        source.connect(this.processor);
        this.processor.connect(this.audioContext.destination);
    }

    stop() {
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
        }
        if (this.processor) {
            this.processor.disconnect();
            this.processor = null;
        }
        if (this.audioContext && this.audioContext.state !== 'closed') {
            this.audioContext.close();
        }
    }
}

// --- IMPORTANT : 4. 메인 애플리케이션 로직 (⭐) ---
const connectBtn = document.getElementById('connectBtn');
const disconnectBtn = document.getElementById('disconnectBtn');
const statusDiv = document.getElementById('status');
const transcriptsDiv = document.getElementById('transcripts');

// 🔧 서버 엔드포인트 변경
const SERVER_URL = "ws://localhost:8765/ws/realtime";
const SEND_SAMPLE_RATE = 16000;
const RECEIVE_SAMPLE_RATE = 24000;

let geminiApi, microphone, audioPlayer, accessToken = null;

// 🆕 Spring 서버에서 JWT 토큰 획득 함수
function getAccessTokenFromUser() {
    // Spring 서버에서 발급받은 JWT 토큰을 입력받음
    const token = prompt('Spring 서버에서 발급받은 JWT 토큰을 입력하세요:');
    return token && token.trim() !== '' ? token.trim() : null;
}

// 🆕 API 상태 확인 함수 추가
async function checkServerHealth() {
    try {
        const response = await fetch('http://localhost:8765/health');
        const data = await response.json();
        console.log('서버 상태:', data);
        return data.status === 'healthy';
    } catch (error) {
        console.error('서버 상태 확인 실패:', error);
        return false;
    }
}

connectBtn.addEventListener('click', async () => {
    connectBtn.disabled = true;
    statusDiv.textContent = '서버 상태 확인 중...';

    // 🆕 서버 헬스 체크
    const isServerHealthy = await checkServerHealth();
    if (!isServerHealthy) {
        updateStatus('❌ 서버가 응답하지 않습니다', '#f8d7da');
        connectBtn.disabled = false;
        return;
    }

    statusDiv.textContent = 'JWT 토큰 입력 대기 중...';
    
    // 🆕 Spring 서버에서 발급받은 JWT 토큰 입력
    accessToken = getAccessTokenFromUser();
    if (!accessToken) {
        updateStatus('❌ JWT 토큰이 필요합니다', '#f8d7da');
        connectBtn.disabled = false;
        return;
    }

    statusDiv.textContent = '연결 중...';
    transcriptsDiv.innerHTML = '';

    audioPlayer = new StreamingAudioPlayer(RECEIVE_SAMPLE_RATE);
    geminiApi = new GeminiAPI(SERVER_URL, accessToken); // 토큰과 함께 API 생성
    setupApiCallbacks();

    microphone = new Microphone(SEND_SAMPLE_RATE, (audioBuffer) => {
        geminiApi.sendAudio(audioBuffer);
    });

    try {
        geminiApi.connect();
        await microphone.start();
    } catch (error) {
        console.error("연결 또는 마이크 시작 중 오류:", error);
        updateStatus(`❌ 오류: ${error.message}`, '#f8d7da');
        stopAll();
    }
});

disconnectBtn.addEventListener('click', () => {
    geminiApi.close();
});

function setupApiCallbacks() {
    geminiApi.onOpen = () => {
        updateStatus('✅ 연결됨 및 녹음 중...', '#d4edda');
        disconnectBtn.disabled = false;
    };
    
    geminiApi.onClose = (event) => {
        // 🔧 연결 종료 이유에 따른 메시지 개선
        const reason = event.reason || '알 수 없는 이유';
        const code = event.code || '알 수 없음';
        console.log(`연결 종료: 코드 ${code}, 이유: ${reason}`);
        
        if (code === 1008) {
            updateStatus('❌ 인증 실패로 연결이 끊어졌습니다', '#f8d7da');
        } else if (code === 1011) {
            updateStatus('❌ 서버 내부 오류로 연결이 끊어졌습니다', '#f8d7da');
        } else {
            updateStatus(`🔌 연결 끊김 (${reason})`, '#fff3cd');
        }
        stopAll();
    };
    
    geminiApi.onError = (error) => {
        console.error('WebSocket 오류:', error);
        updateStatus('❌ 웹소켓 오류 발생', '#f8d7da');
        stopAll();
    };
    
    geminiApi.onAudio = (base64) => audioPlayer.receiveAudio(base64);
    geminiApi.onInputTranscript = (text) => appendTranscript(text, '사용자', 'user-transcript');
    geminiApi.onOutputTranscript = (text) => appendTranscript(text, 'AI', 'ai-transcript');
    
    geminiApi.onTurnComplete = () => {
        console.log("대화 턴 완료.");
        const lastElement = transcriptsDiv.lastElementChild;
        if(lastElement) lastElement.dataset.final = "true";
    };
    
    // Interrupt 콜백 설정
    geminiApi.onInterrupt = () => {
        console.log("오디오 중단 처리");
        audioPlayer.interrupt();
    };
}

function stopAll() {
    microphone?.stop();
    audioPlayer?.stop();
    connectBtn.disabled = false;
    disconnectBtn.disabled = true;
}

function updateStatus(message, color) {
    statusDiv.textContent = message;
    statusDiv.style.backgroundColor = color;
}

function appendTranscript(text, speaker, className) {
    const lastElement = transcriptsDiv.lastElementChild;
    if (lastElement && lastElement.className.includes(className) && lastElement.dataset.final !== "true") {
        const textNode = lastElement.querySelector('.text');
        if(textNode) textNode.textContent += text;
    } else {
         if(lastElement) lastElement.dataset.final = "true";
         const wrapper = document.createElement('div');
         wrapper.className = `transcript-wrapper ${className}`;
         wrapper.innerHTML = `<span class="transcript-label">${speaker}</span><p class="text">${text}</p>`;
         transcriptsDiv.appendChild(wrapper);
    }
    transcriptsDiv.scrollTop = transcriptsDiv.scrollHeight;
}

// 🆕 페이지 로드 시 서버 상태 확인
document.addEventListener('DOMContentLoaded', async () => {
    const isHealthy = await checkServerHealth();
    if (isHealthy) {
        updateStatus('🟢 서버 준비됨 - 연결 가능', '#d1ecf1');
    } else {
        updateStatus('🔴 서버 연결 불가', '#f8d7da');
    }
});