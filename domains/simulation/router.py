# domains/simulation/router.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from openai import OpenAI, OpenAIError
import json
import os
import re

router = APIRouter()

# OpenAI API 키 설정 및 검증
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("경고: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
    client = None
else:
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        print(f"OpenAI 클라이언트 초기화 실패: {e}")
        client = None

# Pydantic 모델 정의
class HintsRequest(BaseModel):
    scenarios: List[Dict[str, Any]]
    task: str

class EducationalAnalysisRequest(BaseModel):
    userid: str
    companyid: int
    userorder: List[int]  
    reason: str
    responseTexts: Dict[str, str] 
    hints: Dict[str, str] 
    orderSelectionTime: int
    reasonWritingTime: int
    totalTimeSpent: int
    
    # ✅ 추가: 시나리오 정보
    scenarioContents: Dict[str, str] = {}  
    scenarioTags: Dict[str, str] = {}      

# 무의미한 입력 검증 함수
def validate_response_text(text: str) -> bool:
    """응답 텍스트 유효성 검증"""
    if not text or len(text.strip()) < 5:
        return False
    
    text_stripped = text.strip()
    
    # 자음/모음만 있는 경우 (ㄱ, ㅏ, ㅀ 등)
    korean_consonants = re.compile(r'^[ㄱ-ㅎㅏ-ㅣ]+$')
    if korean_consonants.match(text_stripped):
        return False
    
    # 반복 문자 (ㅋㅋㅋ, ㅎㅎㅎ, 111, aaa 등)
    repeated_char = re.compile(r'^(.)\1{2,}$')
    if repeated_char.match(text_stripped):
        return False
    
    # 숫자만 있는 경우
    if text_stripped.isdigit():
        return False
    
    # 특수문자만 있는 경우
    if re.match(r'^[^a-zA-Z0-9가-힣]+$', text_stripped):
        return False
    
    # 의미없는 짧은 반복 (ㄱㄱㄱ, 안안안, 네네네 등)
    if len(text_stripped) <= 10:
        # 같은 문자가 50% 이상인 경우
        char_count = {}
        for char in text_stripped:
            char_count[char] = char_count.get(char, 0) + 1
        
        max_char_count = max(char_count.values())
        if max_char_count / len(text_stripped) > 0.5:
            return False
    
    return True

def get_invalid_responses(request: EducationalAnalysisRequest) -> List[str]:
    """무의미한 응답이 있는 시나리오 ID 목록 반환"""
    invalid_responses = []
    for scenario_id, text in request.responseTexts.items():
        if not validate_response_text(text):
            invalid_responses.append(scenario_id)
    return invalid_responses

# 시나리오 내용 매핑 함수 - ✅ 수정된 버전
def get_scenario_info_by_id(scenario_id: int, request: EducationalAnalysisRequest = None) -> Dict[str, str]:
    """실제 시나리오 ID를 받아서 내용과 태그 반환"""
    
    # ✅ 요청에서 실제 시나리오 정보가 있으면 사용
    if request and request.scenarioContents:
        scenario_id_str = str(scenario_id)
        if scenario_id_str in request.scenarioContents:
            return {
                "content": request.scenarioContents[scenario_id_str],
                "tags": request.scenarioTags.get(scenario_id_str, "")
            }
    
    # ❌ 기존 추측 로직 (백업용으로만 사용)
    scenario_patterns = {
        "coffee": {"content": "커피머신이 작동하지 않음", "tags": "출근조,기기고장"},
        "cancel": {"content": "고객이 음료 주문을 취소하겠다고 함", "tags": "고객클레임,주문관리"},
        "newbie": {"content": "신입 직원이 계산을 틀려서 당황함", "tags": "신입교육,실수처리"},
        "waiting": {"content": "매장에 고객이 줄을 서서 대기 중", "tags": "혼잡상황,대기관리"},
        "delivery": {"content": "배달 주문이 5건 동시에 들어옴", "tags": "배달,다중업무"}
    }
    
    patterns = list(scenario_patterns.keys())
    pattern_index = (scenario_id % len(patterns))
    pattern_key = patterns[pattern_index]
    
    return scenario_patterns[pattern_key]

def build_scenarios_info_from_request(request: EducationalAnalysisRequest) -> str:
    """요청에서 받은 시나리오 ID들을 기반으로 1~5 순서의 시나리오 정보 생성"""
    scenarios_info = ""
    
    for i, scenario_id in enumerate(request.userorder, 1):
        # ✅ 수정: request를 함께 전달
        scenario_info = get_scenario_info_by_id(scenario_id, request)
        scenarios_info += f"{i}. {scenario_info['content']} ({scenario_info['tags']})\n"
    
    return scenarios_info

# 힌트 생성 엔드포인트
@router.post("/get-hints")
async def get_hints_only(request: HintsRequest):
    """페이지 로딩 시 각 상황별 응대 힌트만 생성"""
    
    if not client:
        print("OpenAI 클라이언트가 설정되지 않음 - 기본 힌트 반환")
        return create_default_hints(request.scenarios)
    
    scenarios = request.scenarios
    task = request.task
    
    print(f"힌트 요청 받음: {len(scenarios)}개 시나리오")
    
    if not scenarios:
        raise HTTPException(status_code=400, detail="시나리오가 제공되지 않았습니다.")
    
    if task != "generate_hints_only":
        raise HTTPException(status_code=400, detail="잘못된 작업 유형입니다.")
    
    try:
        prompt = generate_hints_only_prompt(scenarios)
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        gpt_response = response.choices[0].message.content
        print(f"GPT 힌트 응답: {gpt_response}")
        
        hints_result = json.loads(gpt_response)
        return hints_result
        
    except OpenAIError as e:
        print(f"❌ OpenAI API 오류: {e}")
        return create_default_hints(scenarios)
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON 파싱 오류: {e}")
        print(f"파싱 실패한 응답: {gpt_response}")
        return create_default_hints(scenarios)
        
    except Exception as e:
        print(f"❌ 힌트 생성 오류: {e}")
        return create_default_hints(scenarios)

# 교육용 분석 엔드포인트
@router.post("/educational-analysis")
async def educational_analysis(request: EducationalAnalysisRequest):
    """교육 중심 시뮬레이션 분석 - 실제 시나리오 ID 지원"""
    
    if not client:
        print("❌ OpenAI 클라이언트가 설정되지 않음")
        return create_default_educational_analysis(request)
    
    print(f"교육용 분석 요청 - 사용자: {request.userid}")
    print(f"선택한 순서 (실제 ID): {request.userorder}")
    print(f"시나리오 정보: {request.scenarioContents}")
    print(f"사용자 이유: {request.reason}")
    
    # 무의미한 입력 검증
    invalid_responses = get_invalid_responses(request)
    if invalid_responses:
        print(f"⚠️ 무의미한 입력 감지 - 시나리오: {invalid_responses}")
    
    try:
        # 1단계: 사용자가 선택한 순서를 기반으로 1~5 시나리오 정보 생성
        print("🔄 시나리오 정보 구성 중...")
        scenarios_info = build_scenarios_info_from_request(request)
        print(f"생성된 시나리오 정보:\n{scenarios_info}")
        
        # 2단계: GPT 추천 순서 및 이유 생성 (1~5 기준)
        print("🔄 GPT 추천 순서 생성 중...")
        gpt_recommendation = await get_gpt_recommended_order_detailed(scenarios_info)
        
        # 3단계: 교육용 분석 프롬프트 생성
        print("🔄 교육용 분석 프롬프트 생성 중...")
        educational_prompt = generate_educational_analysis_prompt(request, gpt_recommendation, invalid_responses)
        
        # GPT API 호출
        print("🔄 GPT API 호출 중...")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=4000
        )
        
        gpt_response = response.choices[0].message.content
        print(f"✅ GPT 교육 분석 응답 길이: {len(gpt_response)}")
        print(f"GPT 응답 미리보기: {gpt_response[:200]}...")
        
        # JSON 파싱 시도
        try:
            analysis_result = json.loads(gpt_response)
            print("✅ JSON 파싱 성공")
        except json.JSONDecodeError:
            # JSON 추출 시도
            print("⚠️ 직접 JSON 파싱 실패, JSON 블록 추출 시도...")
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', gpt_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                analysis_result = json.loads(json_str)
                print("✅ JSON 블록 추출 및 파싱 성공")
            else:
                print("❌ JSON 블록 추출 실패, 기본값 사용")
                return create_improved_default_educational_analysis(request, invalid_responses)
        
        # 사용자 순서 정보 추가 (실제 시나리오 ID 사용)
        analysis_result["userOrder"] = format_user_order_with_real_ids(request.userorder, request)
        
        return analysis_result
        
    except OpenAIError as e:
        print(f"❌ OpenAI API 오류: {e}")
        return create_improved_default_educational_analysis(request, invalid_responses)
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON 파싱 오류: {e}")
        print(f"파싱 실패한 응답: {gpt_response if 'gpt_response' in locals() else 'N/A'}")
        return create_improved_default_educational_analysis(request, invalid_responses)
        
    except Exception as e:
        print(f"❌ 교육 분석 오류: {e}")
        return create_improved_default_educational_analysis(request, invalid_responses)

# 힌트 관련 함수들
def generate_hints_only_prompt(scenarios):
    """힌트 전용 프롬프트 생성"""
    
    prompt = """다음은 카페에서 발생할 수 있는 상황들입니다. 각 상황에 대한 실무에서 바로 사용할 수 있는 고객 응대 힌트를 제공해주세요.

【상황 목록】
"""
    
    for i, scenario in enumerate(scenarios):
        scenario_id = scenario.get('scenarioId')
        content = scenario.get('scenarioContent', '')
        tags = scenario.get('scenarioTag', '')
        prompt += f"{scenario_id}. {content} (태그: {tags})\n"
    
    prompt += """
각 상황별로 현실적이고 실용적인 고객 응대 힌트를 제공해주세요.
태그에 맞는 상황을 고려하여 구체적인 멘트나 행동 지침을 포함해주세요.

다음 JSON 형식으로만 응답해주세요:
{
    "responseHints": {
"""
     
    for i, scenario in enumerate(scenarios):
        scenario_id = scenario.get('scenarioId')
        if i == len(scenarios) - 1:
            prompt += f'        "{scenario_id}": "이 상황에 대한 구체적인 응대 힌트"\n'
        else:
            prompt += f'        "{scenario_id}": "이 상황에 대한 구체적인 응대 힌트",\n'
    
    prompt += """    }
}

반드시 위의 JSON 형식으로만 응답하고, 다른 설명이나 텍스트는 포함하지 마세요.
각 힌트는 실무에서 바로 사용할 수 있도록 구체적이고 실용적으로 작성해주세요."""

    return prompt

def create_default_hints(scenarios):
    """GPT 호출 실패 시 기본 힌트 반환"""
    default_hints = {}
    
    for scenario in scenarios:
        scenario_id = str(scenario.get('scenarioId'))
        content = scenario.get('scenarioContent', '')
        
        if '커피머신' in content or '기기' in content:
            default_hints[scenario_id] = "기기 상태를 확인하고, 필요시 전문가에게 연락하세요."
        elif '고객' in content and ('클레임' in content or '취소' in content):
            default_hints[scenario_id] = "고객의 말씀을 주의깊게 듣고, 진심으로 사과드리며 해결방안을 제시하세요."
        elif '신입' in content or '계산' in content:
            default_hints[scenario_id] = "친절하게 정확한 방법을 알려주고, 실수를 격려의 기회로 만드세요."
        elif '대기' in content or '줄' in content:
            default_hints[scenario_id] = "고객에게 상황을 설명하고, 대기시간을 최소화하도록 노력하세요."
        elif '배달' in content:
            default_hints[scenario_id] = "주문을 정확히 확인하고, 효율적인 순서로 준비하세요."
        else:
            default_hints[scenario_id] = "상황에 맞는 적절한 대응을 하세요."
    
    return {"responseHints": default_hints}

# 교육 분석 관련 함수들
async def get_gpt_recommended_order_detailed(scenarios_info: str):
    """GPT가 추천하는 처리 순서 및 상세 이유 - 1~5 기준 사용"""
    
    prompt = f"""
    카페에서 다음 5가지 상황이 동시에 발생했습니다:
    {scenarios_info}
    
    전문적인 카페 매니저 관점에서 이 상황들을 처리할 최적의 우선순위를 정하고, 
    우선순위 판단 기준과 상세한 이유를 설명해주세요.
    
    **중요: 추천 순서는 반드시 [1, 2, 3, 4, 5] 숫자로 배열해주세요.**
    (1번이 첫 번째 상황, 2번이 두 번째 상황, 3번이 세 번째 상황, 4번이 네 번째 상황, 5번이 다섯 번째 상황)
    
    다음 JSON 형식으로만 응답해주세요:
    {{
        "recommendedOrder": [1, 2, 3, 4, 5 중 순서 배열],
        "priorityCriteria": "우선순위를 정하는 판단 기준 (1-2문장)",
        "detailedReasoning": "우선순위를 이렇게 정한 상세한 이유"
    }}
    """
    
    try:
        if not client:
            raise Exception("OpenAI 클라이언트가 없음")
            
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        gpt_response = response.choices[0].message.content
        return json.loads(gpt_response)
        
    except Exception as e:
        print(f"❌ GPT 추천 순서 생성 실패: {e}")
        return {
            "recommendedOrder": [1, 4, 2, 5, 3],
            "priorityCriteria": "매장 운영에 미치는 파급효과와 고객이 직접 체감하는 불편함의 시급성을 종합적으로 고려합니다.",
            "detailedReasoning": "커피머신 고장은 매장의 핵심 기능에 영향을 미치므로 최우선으로 처리해야 합니다. 대기 고객은 즉시 보이는 문제이므로 두 번째로 처리하고, 주문 취소는 개별 고객 만족도에 직접 영향을 미치므로 세 번째입니다. 배달 주문은 여러 고객이 기다리고 있어 네 번째로 처리하며, 신입 직원 교육은 장기적 관점에서 중요하지만 즉시성이 상대적으로 낮아 마지막으로 처리합니다."
        }

def analyze_time_data(request: EducationalAnalysisRequest) -> str:
    """시간 데이터 분석 - 5초 이하일 때만 참여도 부족 판정"""
    order_time = request.orderSelectionTime
    reason_time = request.reasonWritingTime
    total_time = request.totalTimeSpent
    
    analysis = f"순서 선택 시간: {order_time}초, 이유 작성 시간: {reason_time}초, 전체 소요 시간: {total_time}초\n"
    
    # 순서 선택 시간 분석 (5초 이하만 부족 판정)
    if order_time <= 5:  # 5초 이하
        analysis += "순서 선택이 너무 빨랐습니다. 조금 더 신중하게 고민해보시면 좋겠어요. "
    elif order_time <= 30:  # 30초 이하
        analysis += "순서 선택을 빠르게 완료하셨네요. "
    elif order_time <= 90:  # 1분 30초 이하
        analysis += "순서 선택을 적절한 시간 내에 완료하셨습니다. "
    elif order_time <= 180:  # 3분 이하
        analysis += "순서 선택에 충분한 시간을 투자하셨습니다. "
    else:  # 3분 초과
        analysis += "순서 선택을 매우 신중하게 하셨네요. "
    
    # 이유 작성 시간 분석 (5초 이하만 부족 판정)
    if reason_time <= 5:  # 5초 이하
        analysis += "이유 작성이 너무 빨랐습니다. 조금 더 자세하게 생각해보시면 좋겠어요. "
    elif reason_time <= 60:  # 1분 이하
        analysis += "이유 작성을 빠르게 완료하셨네요. "
    elif reason_time <= 120:  # 2분 이하
        analysis += "이유 작성에 적절한 시간을 사용하셨습니다. "
    elif reason_time <= 300:  # 5분 이하
        analysis += "이유 작성에 충분한 시간을 투자하셨습니다. "
    else:  # 5분 초과
        analysis += "이유 작성을 매우 꼼꼼하게 하셨네요. "
    
    return analysis

def analyze_text_quality(request: EducationalAnalysisRequest, invalid_responses: List[str]) -> str:
    """텍스트 품질 분석 - 5자 이하일 때만 참여도 부족 판정"""
    reason_length = len(request.reason.strip())
    
    total_response_length = sum(len(text.strip()) for text in request.responseTexts.values())
    avg_response_length = total_response_length / len(request.responseTexts) if request.responseTexts else 0
    
    analysis = f"이유 작성 길이: {reason_length}자, 평균 응대 멘트 길이: {avg_response_length:.1f}자\n"
    
    # 무의미한 입력 체크
    if invalid_responses:
        analysis += f"시나리오 {', '.join(invalid_responses)}의 멘트가 무의미한 입력으로 보입니다. "
    
    # 이유 작성 길이 분석 (5자 이하만 부족 판정)
    if reason_length <= 5:  # 5자 이하
        analysis += "순서 선택 이유가 너무 간단합니다. 조금 더 자세한 설명을 추가해주시면 좋겠어요. "
    elif reason_length <= 30:  # 30자 이하
        analysis += "순서 선택 이유를 간단하게 작성하셨네요. "
    elif reason_length <= 100:  # 100자 이하
        analysis += "순서 선택 이유를 적절하게 작성하셨습니다. "
    elif reason_length <= 300:  # 300자 이하
        analysis += "순서 선택 이유를 상세하게 작성하셨네요. "
    else:  # 300자 초과
        analysis += "순서 선택 이유를 매우 자세하게 작성해주셨습니다. "
    
    # 응대 멘트 길이 분석 (5자 이하만 부족 판정)
    if avg_response_length <= 5:  # 5자 이하 (안녕하세요 수준)
        analysis += "응대 멘트가 너무 간단합니다. 좀 더 구체적인 표현을 추가해주시면 좋겠어요. "
    elif avg_response_length <= 20:  # 20자 이하
        analysis += "응대 멘트를 간단하게 작성하셨네요. "
    elif avg_response_length <= 50:  # 50자 이하
        analysis += "응대 멘트를 적절하게 작성하셨습니다. "
    elif avg_response_length <= 100:  # 100자 이하
        analysis += "응대 멘트를 상세하게 작성하셨네요. "
    else:  # 100자 초과
        analysis += "응대 멘트를 매우 구체적으로 작성해주셨습니다. "
    
    return analysis

def generate_educational_analysis_prompt(request: EducationalAnalysisRequest, gpt_recommendation: dict, invalid_responses: List[str]):
    """교육용 분석 프롬프트 생성 - 실제 시나리오 ID 지원"""
    
    # 시간 분석
    time_analysis = analyze_time_data(request)
    
    # 텍스트 길이 분석
    text_analysis = analyze_text_quality(request, invalid_responses)
    
    # 사용자가 선택한 순서를 텍스트로 변환 (1~5 순서 기준)
    user_order_text = ""
    for i, scenario_id in enumerate(request.userorder):
        # ✅ 수정: request를 함께 전달
        scenario_info = get_scenario_info_by_id(scenario_id, request)
        user_order_text += f"{i+1}순위: {scenario_info['content']} (ID: {scenario_id})\n"
    
    # GPT 추천 순서를 텍스트로 변환 (1~5 순서 기준)
    gpt_order_text = ""
    for i, position in enumerate(gpt_recommendation.get('recommendedOrder', [])):
        # position은 1~5 중 하나, 이는 사용자 선택 순서의 인덱스를 의미
        if position <= len(request.userorder):
            actual_scenario_id = request.userorder[position - 1]
            # ✅ 수정: request를 함께 전달
            scenario_info = get_scenario_info_by_id(actual_scenario_id, request)
            gpt_order_text += f"{i+1}순위: {scenario_info['content']} (ID: {actual_scenario_id})\n"
    
    # 각 시나리오별 멘트 (실제 ID 기준)
    response_texts = ""
    for i, scenario_id in enumerate(request.userorder, 1):
        # ✅ 수정: request를 함께 전달
        scenario_info = get_scenario_info_by_id(scenario_id, request)
        response_text = request.responseTexts.get(str(scenario_id), "")
        
        # 무의미한 입력인지 체크
        is_invalid = str(scenario_id) in invalid_responses
        invalid_note = " [⚠️ 무의미한 입력]" if is_invalid else ""
        
        response_texts += f"시나리오 {i} (ID: {scenario_id}) - {scenario_info['content']}{invalid_note}\n사용자 멘트: '{response_text}'\n\n"
    
    # 무의미한 입력에 대한 특별 지침
    invalid_guidance = ""
    if invalid_responses:
        invalid_guidance = f"""
    
    【⚠️ 무의미한 입력 감지】
    시나리오 {', '.join(invalid_responses)}에서 무의미한 입력이 감지되었습니다.
    해당 시나리오들에 대해서는 "이 시나리오에 대해 의미있는 응답을 작성해주시지 않으셨네요. 실제 고객 상황에서 사용할 수 있는 구체적인 멘트를 작성해보시면 어떨까요?"라는 안내를 포함해주세요.
    """
    
    prompt = f"""
    카페 시뮬레이션 교육용 분석을 수행해주세요. 점수나 등급 없이 순수 교육적 관점에서 피드백해주세요.

    【GPT 추천 순서】
    {gpt_order_text}
    판단 기준: {gpt_recommendation.get('priorityCriteria', '')}

    【사용자 선택 순서】
    {user_order_text}
    선택 이유: {request.reason}

    【사용자 응대 멘트】
    {response_texts}

    【시간 및 참여도 분석】
    {time_analysis}
    {text_analysis}
    {invalid_guidance}

    사용자가 실제로 작성한 멘트를 기반으로 개별적이고 구체적인 코칭을 제공해주세요.
    각 시나리오별로 사용자의 실제 멘트 내용을 분석하여 맞춤형 개선점을 제시해야 합니다.

    다음 JSON 형식으로만 응답해주세요:
    {{
        "participationFeedback": "학습 참여도에 대한 피드백 (시간과 텍스트 길이, 무의미한 입력 고려)",
        "scenarioCoaching": {{
            "scenario1": ["사용자 멘트에 대한 구체적 개선점 1", "사용자 멘트에 대한 구체적 개선점 2"],
            "scenario2": ["사용자 멘트에 대한 구체적 개선점 1", "사용자 멘트에 대한 구체적 개선점 2"],
            "scenario3": ["사용자 멘트에 대한 구체적 개선점 1", "사용자 멘트에 대한 구체적 개선점 2"],
            "scenario4": ["사용자 멘트에 대한 구체적 개선점 1", "사용자 멘트에 대한 구체적 개선점 2"],
            "scenario5": ["사용자 멘트에 대한 구체적 개선점 1", "사용자 멘트에 대한 구체적 개선점 2"]
        }},
        "orderAnalysis": "사용자 순서에 대한 분석과 개선 방향을 자연스러운 글로 작성",
        "strengths": [
            "강점 1에 대한 상세 설명",
            "강점 2에 대한 상세 설명", 
            "강점 3에 대한 상세 설명"
        ],
        "learningDirections": [
            "학습 방향 1에 대한 구체적 설명",
            "학습 방향 2에 대한 구체적 설명",
            "학습 방향 3에 대한 구체적 설명"
        ],
        "gptOrderDetails": {{
            "recommendedOrder": [1, 2, 3, 4, 5 순서 배열],
            "formattedOrderList": [
                "1순위: 시나리오명",
                "2순위: 시나리오명",
                "3순위: 시나리오명",
                "4순위: 시나리오명",
                "5순위: 시나리오명"
            ]
        }},
        "gptReasoningDetails": {{
            "priorityCriteria": "GPT가 순서를 정한 판단 기준",
            "detailedReasoning": "GPT가 이런 순서를 추천하는 상세한 이유와 각 순위별 판단 근거"
        }}
    }}

    **중요**: 각 시나리오 코칭은 반드시 사용자가 실제로 작성한 멘트 내용을 분석하여 개별적으로 작성해야 합니다.
    동일한 피드백을 여러 시나리오에 반복 사용하지 마세요.
    무의미한 입력이 감지된 시나리오에는 적절한 안내 메시지를 포함해주세요.
    코칭 톤: "~하시면 더 좋을 것 같아요", "~해보시는 건 어떨까요" 같은 부드럽고 제안하는 말투
    """
    
    return prompt

def format_user_order_with_real_ids(userorder: List[int], request: EducationalAnalysisRequest = None) -> dict:
    """실제 시나리오 ID를 사용한 사용자 순서 정보 포맷팅"""
    
    formatted_order = []
    for i, scenario_id in enumerate(userorder):
        # ✅ 수정: request를 함께 전달
        scenario_info = get_scenario_info_by_id(scenario_id, request)
        formatted_order.append({
            "priority": i + 1,
            "scenarioId": scenario_id,
            "scenarioName": scenario_info['content']
        })
    
    return {
        "order": userorder,
        "formattedOrder": formatted_order
    }

def create_improved_default_educational_analysis(request: EducationalAnalysisRequest, invalid_responses: List[str] = None):
    """개선된 기본 교육 분석 결과 반환 - 실제 시나리오 ID 지원"""
    
    if invalid_responses is None:
        invalid_responses = get_invalid_responses(request)
    
    print("⚠️ GPT 분석 실패 - 개선된 기본 분석 사용")
    
    # 참여도 체크 (5초, 5자 기준)
    order_time = request.orderSelectionTime
    reason_time = request.reasonWritingTime
    reason_length = len(request.reason.strip())
    
    # 참여도 피드백 (5초, 5자 이하일 때만 부족 판정)
    low_participation = (order_time <= 5 or reason_time <= 5 or reason_length <= 5)
    has_invalid_responses = len(invalid_responses) > 0
    
    if low_participation or has_invalid_responses:
        participation_feedback = "교육 효과를 높이기 위해서는 좀 더 신중하게 생각해보시고 상세하게 작성해주시면 좋겠어요. "
        if has_invalid_responses:
            participation_feedback += f"특히 시나리오 {', '.join(invalid_responses)}에서는 의미있는 응답을 작성해주시는 것이 중요합니다. "
        participation_feedback += "더 깊이 있는 학습을 위해 다음에는 시간을 충분히 가지고 참여해보세요."
    else:
        participation_feedback = "교육에 성실하게 참여해주셔서 감사합니다. 이런 자세로 계속 학습해나가시면 좋은 성과를 얻으실 수 있을 거예요."
    
    # 개별 시나리오 코칭 생성 (실제 시나리오 ID 기반)
    coaching = {}
    for i in range(1, 6):
        scenario_key = f"scenario{i}"
        
        # 실제 시나리오 ID 가져오기
        if i <= len(request.userorder):
            actual_scenario_id = request.userorder[i-1]
            user_text = request.responseTexts.get(str(actual_scenario_id), "").strip()
        else:
            user_text = ""
        
        # 무의미한 입력인지 체크
        if str(actual_scenario_id) in invalid_responses:
            coaching[scenario_key] = [
                "작성해주신 멘트가 좋은 방향이지만, 좀 더 구체적으로 표현해보시면 어떨까요?",
                "고객의 입장에서 생각해보시는 것도 도움이 될 것 같아요"
            ]
        elif not user_text:
            coaching[scenario_key] = [
                "작성해주신 멘트가 좋은 방향이지만, 좀 더 구체적으로 표현해보시면 어떨까요?",
                "고객의 입장에서 생각해보시는 것도 도움이 될 것 같아요"
            ]
        elif len(user_text) <= 5:  # 5자 이하만 부족 판정
            coaching[scenario_key] = [
                "작성해주신 멘트가 좋은 방향이지만, 좀 더 구체적으로 표현해보시면 어떨까요?",
                "고객의 입장에서 생각해보시는 것도 도움이 될 것 같아요"
            ]
        else:
            coaching[scenario_key] = [
                "작성해주신 멘트가 좋은 방향이지만, 좀 더 구체적으로 표현해보시면 어떨까요?",
                "고객의 입장에서 생각해보시는 것도 도움이 될 것 같아요"
            ]
    
    # 순서 분석
    gpt_order = [1, 4, 2, 5, 3]  # 기본 추천 순서 (1~5 기준)
    
    # 사용자 순서를 1~5 기준으로 변환하여 비교
    user_order_normalized = list(range(1, len(request.userorder) + 1))
    
    order_analysis = "기본적인 상황 인식 능력을 갖추고 계시지만, 고객 중심적 사고를 조금 더 보강하시면 더욱 균형 잡힌 판단을 하실 수 있을 것 같아요."
    
    # 학습 방향
    learning_directions = [
        "고객 중심 사고 기르기를 연습해보세요",
        "상황별 커뮤니케이션 스킬을 늘려가보세요",
        "체계적인 문제 해결 방법을 배워보시면 좋겠어요"
    ]
    
    if has_invalid_responses:
        learning_directions[1] = "의미있는 문장으로 고객과 소통하는 능력을 기르고, 상황별 커뮤니케이션 스킬을 늘려가보세요"
    
    # GPT 추천 순서를 실제 시나리오 내용으로 변환
    formatted_order_list = []
    for i, position in enumerate(gpt_order, 1):
        if position <= len(request.userorder):
            actual_scenario_id = request.userorder[position - 1]
            # ✅ 수정: request를 함께 전달
            scenario_info = get_scenario_info_by_id(actual_scenario_id, request)
            formatted_order_list.append(f"{i}순위: {scenario_info['content']}")
    
    return {
        "userOrder": format_user_order_with_real_ids(request.userorder, request),
        "participationFeedback": participation_feedback,
        "scenarioCoaching": coaching,
        "orderAnalysis": order_analysis,
        "strengths": [
            "빠른 상황 판단 능력을 갖추고 계세요",
            "직원을 생각하는 따뜻한 마음이 돋보여요",
            "기본적인 고객 서비스 마인드를 갖추고 있어요"
        ],
        "learningDirections": learning_directions,
        "gptOrderDetails": {
            "recommendedOrder": gpt_order,
            "formattedOrderList": formatted_order_list
        },
        "gptReasoningDetails": {
            "priorityCriteria": "매장 운영에 미치는 파급효과와 고객이 직접 체감하는 불편함의 시급성을 종합적으로 고려합니다.",
            "detailedReasoning": "커피머신 고장은 매장의 핵심 기능에 영향을 미치므로 최우선으로 처리해야 합니다. 대기 고객은 즉시 보이는 문제이므로 두 번째로 처리하고, 주문 취소는 개별 고객 만족도에 직접 영향을 미치므로 세 번째입니다. 배달 주문은 여러 고객이 기다리고 있어 네 번째로 처리하며, 신입 직원 교육은 장기적 관점에서 중요하지만 즉시성이 상대적으로 낮아 마지막으로 처리합니다."
        }
    }

def create_default_educational_analysis(request: EducationalAnalysisRequest):
    """구버전 호환성을 위한 기본 함수"""
    return create_improved_default_educational_analysis(request)