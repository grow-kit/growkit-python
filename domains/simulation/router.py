# domains/simulation/router.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from openai import OpenAI, OpenAIError
import json
import os
import re
import logging
import asyncio

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# OpenAI API 키 설정 및 검증
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.warning("경고: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
    client = None
else:
    try:
        client = OpenAI(api_key=api_key)
        logger.info("OpenAI 클라이언트 초기화 성공")
    except Exception as e:
        logger.error(f"OpenAI 클라이언트 초기화 실패: {e}")
        client = None


# =============================================================================
# Pydantic 모델 정의
# =============================================================================

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
    scenarioContents: Dict[str, str] = {}
    scenarioTags: Dict[str, str] = {}


# =============================================================================
# 입력 검증 로직
# =============================================================================

def validate_response_text(text: str) -> bool:
    """응답 텍스트 유효성 검증"""
    if not text or len(text.strip()) < 3:
        return False

    text_stripped = text.strip()

    # 자음/모음만 있는 경우
    korean_consonants = re.compile(r'^[ㄱ-ㅎㅏ-ㅣ\s]+$')
    if korean_consonants.match(text_stripped):
        return False

    # 반복 문자 체크
    repeated_char = re.compile(r'^(.)\1{2,}$')
    if repeated_char.match(text_stripped):
        return False

    # 숫자만 있는 경우
    if text_stripped.replace(' ', '').isdigit():
        return False

    # 특수문자만 있는 경우
    if re.match(r'^[^a-zA-Z0-9가-힣\s]+$', text_stripped):
        return False

    # 의미없는 짧은 반복 체크
    if len(text_stripped) <= 15:
        char_count = {}
        for char in text_stripped.replace(' ', ''):
            char_count[char] = char_count.get(char, 0) + 1

        if char_count:
            max_char_count = max(char_count.values())
            total_chars = len(text_stripped.replace(' ', ''))
            if total_chars > 0 and max_char_count / total_chars > 0.6:
                return False

    # 무의미한 패턴들 체크
    meaningless_patterns = [
        r'^[ㅋㅎ]+$',
        r'^\w{1,2}$',
        r'^[.]+$',
        r'^[-]+$',
        r'^[!@#$%^&*()]+$',
    ]

    for pattern in meaningless_patterns:
        if re.match(pattern, text_stripped):
            return False

    return True


def get_invalid_responses(request: EducationalAnalysisRequest) -> List[str]:
    """무의미한 응답이 있는 시나리오 ID 목록 반환"""
    invalid_responses = []
    for scenario_id, text in request.responseTexts.items():
        if not validate_response_text(text):
            invalid_responses.append(scenario_id)
            logger.info(f"무의미한 응답 감지 - 시나리오 {scenario_id}: '{text}'")
    return invalid_responses


# =============================================================================
# 시나리오 정보 매핑 로직
# =============================================================================

def get_scenario_info_by_id(scenario_id: int, request: EducationalAnalysisRequest = None) -> Dict[str, str]:
    """시나리오 ID를 받아서 내용과 태그 반환"""

    # 실제 요청에서 시나리오 정보가 있으면 사용
    if request and request.scenarioContents:
        scenario_id_str = str(scenario_id)
        if scenario_id_str in request.scenarioContents:
            content = request.scenarioContents[scenario_id_str]
            tags = request.scenarioTags.get(scenario_id_str, "")
            logger.info(f"실제 시나리오 정보 사용 - ID {scenario_id}: {content}")
            return {
                "content": content,
                "tags": tags
            }

    # 시나리오 정보가 없으면 에러 발생
    raise HTTPException(
        status_code=400,
        detail=f"시나리오 ID {scenario_id}에 대한 정보를 찾을 수 없습니다. 시나리오 데이터를 확인해주세요."
    )


def build_scenarios_info_from_request(request: EducationalAnalysisRequest) -> str:
    """시나리오 정보 문자열 생성"""
    scenarios_info = ""

    logger.info(f"시나리오 정보 구성 시작 - 사용자 순서: {request.userorder}")

    for i, scenario_id in enumerate(request.userorder, 1):
        scenario_info = get_scenario_info_by_id(scenario_id, request)
        scenarios_info += f"{i}. {scenario_info['content']} (태그: {scenario_info['tags']})\n"

    logger.info(f"완성된 시나리오 정보:\n{scenarios_info}")
    return scenarios_info


def format_user_order_with_real_ids(userorder: List[int], request: EducationalAnalysisRequest = None) -> dict:
    """사용자 순서 정보 포맷팅"""

    formatted_order = []
    for i, scenario_id in enumerate(userorder):
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


# =============================================================================
# GPT API 호출 및 응답 처리
# =============================================================================

async def get_gpt_recommended_order_detailed(scenarios_info: str) -> Dict[str, Any]:
    """GPT 추천 순서 생성"""

    if not client:
        raise HTTPException(status_code=503, detail="AI 서비스가 설정되지 않았습니다")

    prompt = f"""
카페에서 다음 5가지 상황이 동시에 발생했습니다:

{scenarios_info}

전문적인 카페 매니저 관점에서 이 상황들을 처리할 최적의 우선순위를 정하고, 
우선순위 판단 기준과 상세한 이유를 설명해주세요.

**중요: 추천 순서는 반드시 [1, 2, 3, 4, 5] 숫자로 배열해주세요.**

반드시 다음 JSON 형식으로만 응답해주세요:

{{
    "recommendedOrder": [1, 2, 3, 4, 5],
    "priorityCriteria": "우선순위를 정하는 판단 기준 (한 문장으로 명확하게)",
    "detailedReasoning": "우선순위를 이렇게 정한 상세한 이유 (완전한 문장으로)"
}}
"""

    gpt_response = ""

    try:
        logger.info("GPT 추천 순서 요청 중...")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 카페 운영 전문가입니다. 항상 정확한 JSON 형식으로만 응답하세요."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )

        gpt_response = response.choices[0].message.content.strip()
        logger.info(f"GPT 추천 순서 응답 받음: {len(gpt_response)}자")

        parsed_response = json.loads(gpt_response)

        # 응답 유효성 검증
        if "recommendedOrder" not in parsed_response:
            raise ValueError("recommendedOrder 필드가 없음")

        if not isinstance(parsed_response["recommendedOrder"], list):
            raise ValueError("recommendedOrder가 리스트가 아님")

        if len(parsed_response["recommendedOrder"]) != 5:
            raise ValueError("recommendedOrder의 길이가 5가 아님")

        logger.info("GPT 추천 순서 파싱 성공")
        return parsed_response

    except OpenAIError as e:
        logger.error(f"OpenAI API 오류: {e}")
        raise HTTPException(status_code=502, detail="AI 서비스 연결에 문제가 있습니다")
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 실패: {e}")
        if gpt_response:
            logger.error(f"응답 내용: {gpt_response}")
        raise HTTPException(status_code=502, detail="AI 응답 형식에 문제가 있습니다")
    except Exception as e:
        logger.error(f"추천 순서 생성 실패: {e}")
        if gpt_response:
            logger.error(f"응답 내용: {gpt_response}")
        raise HTTPException(status_code=500, detail="추천 순서 생성 중 오류가 발생했습니다")


def parse_gpt_response_safely(gpt_response: str) -> Optional[Dict[str, Any]]:
    """GPT 응답 안전 파싱"""

    if not gpt_response or not gpt_response.strip():
        logger.error("GPT 응답이 비어있음")
        return None

    logger.info(f"파싱할 GPT 응답 길이: {len(gpt_response)}")
    logger.info(f"GPT 응답 시작 부분: {gpt_response[:200]}")

    try:
        # 1단계: 직접 JSON 파싱
        result = json.loads(gpt_response.strip())
        logger.info("✅ 직접 JSON 파싱 성공")
        return result

    except json.JSONDecodeError as e:
        logger.warning(f"직접 JSON 파싱 실패: {e}")

        # 응답 전체를 로그로 출력 (디버깅용)
        logger.error(f"파싱 실패한 전체 응답:\n{gpt_response}")

        try:
            # 2단계: JSON 코드 블록 추출
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', gpt_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                logger.info("JSON 코드 블록 발견, 파싱 시도")
                return json.loads(json_str)

            # 3단계: 첫 번째와 마지막 중괄호 사이 추출
            first_brace = gpt_response.find('{')
            last_brace = gpt_response.rfind('}')

            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_candidate = gpt_response[first_brace:last_brace + 1]
                logger.info(f"중괄호 구간 추출: {len(json_candidate)}자")
                logger.info(f"추출된 JSON: {json_candidate[:300]}...")
                return json.loads(json_candidate)

        except json.JSONDecodeError as e2:
            logger.error(f"모든 파싱 시도 실패: {e2}")
            return None

    except Exception as e:
        logger.error(f"예상치 못한 파싱 오류: {e}")
        return None


# =============================================================================
# 교육 분석 프롬프트 및 로직
# =============================================================================

def generate_educational_analysis_prompt(request: EducationalAnalysisRequest, gpt_recommendation: dict,
                                         invalid_responses: List[str] = None):
    """교육용 분석 프롬프트 생성 - 명확한 JSON 구조 요구"""

    if invalid_responses is None:
        invalid_responses = get_invalid_responses(request)

    # 사용자 응답 분석
    user_responses_analysis = ""
    for i, scenario_id in enumerate(request.userorder, 1):
        scenario_info = get_scenario_info_by_id(scenario_id, request)
        user_text = request.responseTexts.get(str(scenario_id), "").strip()
        invalid_note = " [⚠️ 무의미한 입력]" if str(scenario_id) in invalid_responses else ""
        user_responses_analysis += f"시나리오 {i}: {scenario_info['content']}{invalid_note}\n사용자 응답: \"{user_text}\"\n\n"

    # 사용자 순서
    user_order_text = ""
    for i, scenario_id in enumerate(request.userorder):
        scenario_info = get_scenario_info_by_id(scenario_id, request)
        user_order_text += f"{i + 1}순위: {scenario_info['content']}\n"

    # GPT 추천 순서
    gpt_order_text = ""
    gpt_order = gpt_recommendation.get('recommendedOrder', [1, 2, 3, 4, 5])
    for i, position in enumerate(gpt_order, 1):
        if position <= len(request.userorder):
            actual_scenario_id = request.userorder[position - 1]
            scenario_info = get_scenario_info_by_id(actual_scenario_id, request)
            gpt_order_text += f"{i}순위: {scenario_info['content']}\n"

    # 안전한 문자열 처리
    priority_criteria = gpt_recommendation.get('priorityCriteria', '').replace('"', "'")
    detailed_reasoning = gpt_recommendation.get('detailedReasoning', '').replace('"', "'")

    prompt = f"""
카페 시뮬레이션 교육 분석을 수행하고 JSON으로 응답해주세요.

=== 사용자 정보 ===
사용자: {request.userid}
순서 선택 시간: {request.orderSelectionTime}초
이유 작성 시간: {request.reasonWritingTime}초
선택 이유: "{request.reason}"

=== 사용자가 선택한 순서 ===
{user_order_text}

=== AI 추천 순서 ===
{gpt_order_text}

=== 사용자 응답들 ===
{user_responses_analysis}

**중요 요구사항:**
1. strengths는 완전한 문장 2개 또는 3개로 작성 (각 문장이 끝나고 콤마로 구분)
2. learningDirections도 완전한 문장 2개 또는 3개로 작성 (각 문장이 끝나고 콤마로 구분)
3. 모든 문장은 "습니다", "세요", "요" 등으로 완전히 끝나야 함
4. JSON 형식을 정확히 지켜주세요

다음과 같은 정확한 JSON 형식으로만 응답하세요:

{{
  "participationFeedback": "참여도에 대한 완전한 문단",
  "scenarioCoaching": {{
    "scenario1": ["개선점1", "개선점2"],
    "scenario2": ["개선점1", "개선점2"],
    "scenario3": ["개선점1", "개선점2"],
    "scenario4": ["개선점1", "개선점2"],
    "scenario5": ["개선점1", "개선점2"]
  }},
  "orderAnalysis": "순서 선택에 대한 완전한 문단",
  "strengths": [
    "완전한 문장으로 된 강점 하나입니다",
    "완전한 문장으로 된 강점 둘입니다"
  ],
  "learningDirections": [
    "완전한 문장으로 된 학습방향 하나입니다",
    "완전한 문장으로 된 학습방향 둘입니다"
  ],
  "gptOrderDetails": {{
    "recommendedOrder": {gpt_recommendation.get('recommendedOrder', [1, 2, 3, 4, 5])},
    "formattedOrderList": [
      "1순위: 첫번째 시나리오",
      "2순위: 두번째 시나리오", 
      "3순위: 세번째 시나리오",
      "4순위: 네번째 시나리오",
      "5순위: 다섯번째 시나리오"
    ]
  }},
  "gptReasoningDetails": {{
    "priorityCriteria": "{priority_criteria}",
    "detailedReasoning": "{detailed_reasoning}"
  }}
}}

반드시 위의 JSON 형식만 사용하고, 다른 텍스트는 포함하지 마세요.
strengths와 learningDirections의 각 항목은 완전한 문장으로 끝나야 합니다.
"""

    return prompt


# =============================================================================
# 메인 엔드포인트들
# =============================================================================

@router.post("/get-hints")
async def get_hints_only(request: HintsRequest):
    """힌트 생성"""

    if not client:
        raise HTTPException(status_code=503, detail="AI 서비스가 설정되지 않았습니다")

    scenarios = request.scenarios
    task = request.task

    logger.info(f"힌트 요청 받음: {len(scenarios)}개 시나리오")

    if not scenarios:
        raise HTTPException(status_code=400, detail="시나리오가 제공되지 않았습니다")

    if task != "generate_hints_only":
        raise HTTPException(status_code=400, detail="잘못된 작업 유형입니다")

    try:
        prompt = generate_hints_only_prompt(scenarios)

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 카페 운영 전문가입니다. 실무에서 바로 사용할 수 있는 구체적인 힌트를 JSON 형식으로만 제공하세요."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )

        gpt_response = response.choices[0].message.content
        logger.info(f"GPT 힌트 응답 길이: {len(gpt_response)}")

        try:
            hints_result = json.loads(gpt_response)
        except json.JSONDecodeError:
            logger.error(f"힌트 JSON 파싱 실패. 응답: {gpt_response}")
            raise HTTPException(status_code=502, detail="AI 응답 형식에 문제가 있습니다")

        # 응답 구조 검증
        if "responseHints" not in hints_result:
            logger.error(f"responseHints 필드 없음. 응답: {hints_result}")
            raise HTTPException(status_code=502, detail="AI 응답 구조에 문제가 있습니다")

        return hints_result

    except OpenAIError as e:
        logger.error(f"OpenAI API 오류: {e}")
        raise HTTPException(status_code=502, detail="AI 서비스 연결에 문제가 있습니다")
    except HTTPException:
        # 이미 적절한 HTTPException이므로 재발생
        raise
    except Exception as e:
        logger.error(f"힌트 생성 예상치 못한 오류: {e}")
        raise HTTPException(status_code=500, detail="힌트 생성에 실패했습니다")


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
힌트는 실제 매장에서 바로 사용할 수 있는 구체적인 멘트나 행동 지침이어야 합니다.

다음 JSON 형식으로만 응답해주세요:
{
    "responseHints": {
"""

    for i, scenario in enumerate(scenarios):
        scenario_id = scenario.get('scenarioId')
        if i == len(scenarios) - 1:
            prompt += f'        "{scenario_id}": "이 상황에 대한 구체적이고 실용적인 응대 힌트"\n'
        else:
            prompt += f'        "{scenario_id}": "이 상황에 대한 구체적이고 실용적인 응대 힌트",\n'

    prompt += """    }
}

반드시 위의 JSON 형식으로만 응답하고, 다른 설명이나 텍스트는 포함하지 마세요.
각 힌트는 50-80자 내외의 구체적이고 실용적인 조언으로 작성해주세요."""

    return prompt


@router.post("/educational-analysis")
async def educational_analysis(request: EducationalAnalysisRequest):
    """교육용 분석"""

    if not client:
        raise HTTPException(status_code=503, detail="AI 분석 서비스가 설정되지 않았습니다")

    logger.info(f"교육용 분석 요청 - 사용자: {request.userid}")
    logger.info(f"선택한 순서: {request.userorder}")

    # 무의미한 입력 검증
    invalid_responses = get_invalid_responses(request)
    if invalid_responses:
        logger.warning(f"무의미한 입력 감지 - 시나리오: {invalid_responses}")

    # 너무 많은 무의미한 응답이 있으면 에러
    if len(invalid_responses) >= 4:
        raise HTTPException(
            status_code=400,
            detail="의미있는 응답을 더 많이 작성해주세요. 현재 분석하기 어려운 상태입니다."
        )

    try:
        # 1단계: 시나리오 정보 구성
        logger.info("🔄 시나리오 정보 구성 중...")
        scenarios_info = build_scenarios_info_from_request(request)

        # 2단계: GPT 추천 순서 생성
        logger.info("🔄 GPT 추천 순서 생성 중...")
        gpt_recommendation = await get_gpt_recommended_order_detailed(scenarios_info)

        # 3단계: 교육용 분석 프롬프트 생성
        logger.info("🔄 교육용 분석 프롬프트 생성 중...")
        educational_prompt = generate_educational_analysis_prompt(request, gpt_recommendation, invalid_responses)

        # 4단계: GPT API 호출
        logger.info("🔄 GPT API 호출 중...")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 카페 교육 전문가입니다. 사용자의 실제 응답을 분석하여 개인화된 피드백을 제공하세요. 강점과 학습방향은 반드시 2-3개만 제공하고, 모든 문장은 완전해야 합니다. JSON 형식으로만 응답하세요."
                },
                {"role": "user", "content": educational_prompt}
            ],
            temperature=0.5,
            max_tokens=4000
        )

        gpt_response = response.choices[0].message.content
        logger.info(f"✅ GPT 교육 분석 응답 길이: {len(gpt_response)}")

        # 5단계: JSON 파싱
        analysis_result = parse_gpt_response_safely(gpt_response)

        if analysis_result is None:
            logger.error("JSON 파싱 완전 실패")
            raise HTTPException(status_code=502, detail="AI 응답 처리 중 오류가 발생했습니다")

        # 6단계: 필수 필드 검증
        required_fields = ["participationFeedback", "scenarioCoaching", "orderAnalysis", "strengths",
                           "learningDirections"]
        for field in required_fields:
            if field not in analysis_result:
                raise HTTPException(status_code=502, detail=f"AI 응답에 {field} 필드가 없습니다")

        # 7단계: 사용자 순서 정보 추가
        analysis_result["userOrder"] = format_user_order_with_real_ids(request.userorder, request)

        logger.info("✅ 교육 분석 완료")
        return analysis_result

    except HTTPException:
        # 이미 적절한 HTTPException이므로 그대로 재발생
        raise
    except OpenAIError as e:
        logger.error(f"OpenAI API 오류: {e}")
        raise HTTPException(status_code=502, detail="AI 분석 서비스 연결에 문제가 있습니다")
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 오류: {e}")
        raise HTTPException(status_code=502, detail="AI 응답 처리 중 오류가 발생했습니다")
    except Exception as e:
        logger.error(f"교육 분석 오류: {e}")
        raise HTTPException(status_code=500, detail="분석 처리 중 오류가 발생했습니다")