# AI 기반 가공매출 탐지 모델 개발

> 공개 거래 데이터의 수치 분포에 가상의 회계 거래 구조와 부정 시나리오를 결합하여, 규칙 기반 위험점수와 머신러닝 스태킹 모델의 탐지 결과를 비교한 **개인 학습 프로젝트**입니다.

> **유의사항:** 실제 기업의 거래자료·감사자료·부정 사례는 사용하지 않았습니다. 코드와 생성 파일에 포함된 기업·거래 관련 명칭은 가상 시나리오 구현을 위한 예시 식별자이며, 해당 기업의 실제 거래나 부정과는 관련이 없습니다. 본 결과는 구성한 데이터와 가정에 한정되며, 모델의 실무 감사 효과나 일반적인 우월성을 입증하지 않습니다.

## English Summary

- **Project:** Learning Project on Fictitious-Sales Screening
- **Objective:** Compare a predefined rule-based risk score with a machine-learning stacking model on a constructed transaction dataset.
- **Scenarios:** N:M split payments with time lags and hypothetical third-party routing.
- **Result:** In the validation sample used for the portfolio, Recall was 24.12% for the rule-based approach and 70.72% for the stacking model.
- **Caution:** The results are specific to the constructed dataset and assumptions and do not establish practical audit effectiveness.
- **Tech Stack:** Python · Pandas · Scikit-learn

---

## 1. 프로젝트 개요

공개 신용카드 거래 데이터의 익명화된 수치 분포를 참고하고, 날짜·거래처·입출금·거래유형 등 회계적 맥락을 추가하여 약 2만 건의 학습용 거래 데이터를 구성했습니다.

이후 사전에 정의한 규칙 기반 위험점수와 머신러닝 스태킹 모델을 동일한 검증 표본에서 평가하고, 결과가 어떤 데이터 구조와 가정에서 도출되었는지와 그 한계를 함께 검토했습니다.

- **수행 기간:** 2025.09.20 ~ 2025.12.12
- **프로젝트 구분:** 개인 학습 프로젝트
- **활용 기술:** Excel · Python · Pandas · Scikit-learn
- **분석 흐름:** 시나리오 설계 → 데이터 구성 → 모델 비교 → 결과 및 한계 검토
- **발표자료:** [포트폴리오 PDF 보기](./가공%20매출%20탐지%20모델%20개발%20포트폴리오.pdf)

---

## 2. 가상 부정 시나리오

### N:M 분할 지급과 시차 교란

여러 건의 가공매출을 기간별로 묶은 뒤, 총액의 일부를 다음 기간부터 불규칙한 금액과 시점으로 나누어 지급하는 구조를 가정했습니다.

단순한 동일 금액·동일 시점의 1:1 매칭만으로는 포착하기 어려운 패턴을 학습용 데이터에 반영하는 것이 목적이었습니다.

### 제3자 우회 지급

리베이트 일부가 원 거래처가 아닌 제3의 가상 계좌로 지급되는 상황을 가정했습니다.

거래 상대방의 변화까지 고려하여 원 거래와 후속 자금 흐름을 연결하는 과정의 복잡성을 학습용 시나리오에 반영했습니다.

---

## 3. 데이터 구성과 분석 변수

공개 거래 데이터의 익명화된 특징과 금액 분포를 참고하고, 다음과 같은 회계적 맥락을 추가했습니다.

- 가상 거래처와 거래유형
- 입금·출금 방향
- 거래 일자와 시간
- 계절성 특성
- 정상 거래, 가공매출, 리베이트 거래
- N:M 분할 지급과 제3자 우회 지급

### 주요 분석 변수

| 변수 | 내용 |
|---|---|
| `Time_Delta_Seconds` | 동일 거래처의 직전 거래와 현재 거래 사이의 시간 간격 |
| `Is_OffHours` | 주말 또는 영업시간 외 거래 여부 |
| `Amount_vs_Avg` | 해당 거래금액이 거래처 평균금액 대비 얼마나 큰지 |
| `Deposit` / `Withdrawal` | 입금과 출금의 방향 및 금액 |

---

## 4. 모델 비교

### 규칙 기반 위험점수

다음 위험지표에 사전 정의한 점수를 부여하고, 일정 점수 이상인 거래를 이상 항목으로 분류했습니다.

- 거래처 평균 대비 거래금액 비율
- 직전 거래와의 시간 간격
- 주말·영업시간 외 거래 여부

### 스태킹 모델

여러 특징의 결합 패턴을 학습하도록 다음 모델을 구성했습니다.

- **기본 모델:** Random Forest, Gradient Boosting
- **최종 분류기:** Logistic Regression
- **학습·검증 분리:** 전체 데이터의 70%를 학습, 30%를 검증에 사용

두 접근법은 동일한 검증 표본에서 평가했습니다. 다만 사용 방식과 모델 구조가 서로 다르므로, 아래 결과는 구성한 학습용 데이터와 검증 조건 안에서의 비교로 해석해야 합니다.

---

## 5. 프로젝트 결과

| 접근법 | Recall |
|---|---:|
| 규칙 기반 위험점수 | 24.12% |
| 스태킹 모델 | 70.72% |
| 차이 | +46.60%p |

위 수치는 포트폴리오 작성 당시의 실행 결과입니다. 데이터 생성 과정에 난수가 포함되어 있어 코드를 다시 실행하면 수치가 일부 달라질 수 있습니다.

해당 검증 조건에서는 스태킹 모델이 설계된 부정 패턴을 더 많이 포착했습니다. 그러나 Recall만으로 실무 적합성을 판단할 수 없으며, 이상 항목으로 분류된 거래는 계약·증빙·내부통제·산업 특성 등 회계적 맥락을 통해 다시 검토해야 합니다.

---

## 6. 주요 한계

- 실제 기업의 거래자료나 감사자료가 아닌 공개·학습용 데이터 사용
- 부정 시나리오와 특징 변수가 작성자의 가정에 영향
- 산업별 거래 특성, 계약조건, 내부통제와 감사증빙 미반영
- 별도의 외부 데이터에 대한 재검증 미수행
- 실무 적용을 위해서는 데이터 구조와 처리 절차, 모델 성능에 대한 추가 검증 필요

---

## 7. 파일 구성

| 파일 | 내용 |
|---|---|
| `01_data_creation.py` | 공개 데이터 다운로드, 가상 거래 구조 및 학습용 데이터 생성 |
| `02_fraud_detection.py` | 분석 변수 처리, 규칙 기반 위험점수와 스태킹 모델 비교 |
| `가공 매출 탐지 모델 개발 포트폴리오.pdf` | 프로젝트 배경, 과정, 결과와 한계를 정리한 발표자료 |

---

## 8. 실행 방법

### Case 1. 데이터 생성부터 실행

1. 저장소를 다운로드하고 프로젝트 폴더 안에 `data` 폴더를 생성합니다.

2. 필요한 라이브러리를 설치합니다.

```bash
pip install pandas scikit-learn
```

3. 데이터 생성 코드를 실행합니다.

```bash
python 01_data_creation.py
```

`creditcard.csv`가 없는 경우 공개 원본 데이터가 자동으로 다운로드되며, 실행 후 `data` 폴더에 다음 파일이 생성됩니다.

- `shinsegae_sales_ledger_Full.csv`
- `shinsegae_bank_statement_For_ML.csv`

위 파일명은 코드상 가상 데이터 식별자이며 실제 해당 기업의 자료가 아닙니다.

4. 분석 코드를 실행합니다.

```bash
python 02_fraud_detection.py
```

### Case 2. 미리 생성한 데이터로 분석만 실행

아래 파일을 다운로드하여 `data` 폴더에 넣은 뒤 분석 코드를 실행합니다.

- [전체 거래원장 데이터](https://drive.google.com/uc?export=download&id=1NRnm4wUN9uLBB6Ip5N8dYe1oGSpi4eis)
- [학습용 입출금 데이터](https://drive.google.com/uc?export=download&id=1WmuKtBbl8r-JOKrclcs2IuUQpVXNEYiB)

저장 파일명은 각각 다음과 같이 유지합니다.

- `shinsegae_sales_ledger_Full.csv`
- `shinsegae_bank_statement_For_ML.csv`

```bash
python 02_fraud_detection.py
```

---

## 9. 핵심 학습

- 모델보다 먼저 데이터의 출처·정의·품질을 확인해야 합니다.
- 이상 항목으로 분류된 이유를 설명할 수 있어야 합니다.
- 성능지표는 데이터와 가정의 한계와 함께 제시해야 합니다.
- 기술은 전문적 판단을 대체하기보다 검토 범위를 좁히는 도구로 활용해야 합니다.
- 분석 절차와 결과를 문서화하여 검토 가능성을 확보해야 합니다.
