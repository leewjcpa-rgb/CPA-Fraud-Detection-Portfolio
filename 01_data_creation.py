import os
import urllib.request
import pandas as pd
import numpy as np

# --- 0단계: 원본 데이터 자동 다운로드 ---
file_name = "creditcard.csv"
file_url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
if not os.path.exists(file_name):
    print(f"'{file_name}' 파일이 존재하지 않습니다. 다운로드를 시작합니다...")
    try:
        urllib.request.urlretrieve(file_url, file_name)
        print("다운로드를 완료했습니다.")
    except Exception as e:
        print(f"다운로드 중 오류가 발생했습니다: {e}")
        exit()
else:
    print(f"'{file_name}' 파일이 이미 존재합니다.")
print("-" * 50)

# ===================================================================================
# Part 1: '신세계 백화점 전체 거래원장' 생성
# ===================================================================================
print("[Part 1] '신세계 백화점 전체 거래원장' 생성을 시작합니다...")
df = pd.read_csv('creditcard.csv')

# --- 1단계: 시간 데이터 변환 ---
df['TransactionDate'] = pd.to_datetime('2024-01-01') + pd.to_timedelta((df['Time'] / df['Time'].max()) * (365 * 24 * 60 * 60), unit='s')
df = df.drop(['Time'], axis=1)

# --- 2단계: 거래처 특성 부여 ---
client_ids = [f'Client_{i:04d}' for i in range(1, 1001)]
vip_clients = client_ids[:50]
normal_clients = client_ids[50:]
client_probabilities = [0.7 / len(vip_clients)] * len(vip_clients) + [0.3 / len(normal_clients)] * len(normal_clients)
df['Client_ID'] = np.random.choice(client_ids, size=len(df), p=client_probabilities)
df.loc[df['Client_ID'].isin(vip_clients), 'Amount'] *= 3

# --- 3단계: 계절성 반영 ---
df['Month'] = df['TransactionDate'].dt.month
peak_season_transactions = df[df['Month'].isin([3, 6, 9, 12]) & (df['TransactionDate'].dt.day >= 24)]
df = pd.concat([df, peak_season_transactions.sample(frac=0.5, random_state=42)])
df.reset_index(drop=True, inplace=True)

# --- 메인 데이터셋 저장 [경로 수정 완료] ---
main_dataset_filename = "data/shinsegae_sales_ledger_Full.csv"
df.to_csv(main_dataset_filename, index=False)
print(f"\n✅ [Part 1] 완료: {len(df):,}건의 전체 거래원장이 '{main_dataset_filename}'으로 저장되었습니다.")
print("-" * 50)

# ===================================================================================
# Part 2: '회계부정 학습용 입출금 내역' 생성
# ===================================================================================
print("[Part 2] '회계부정 학습용 데이터셋' 생성을 시작합니다...")

# --- 4단계: 부정 거래 및 정상 거래 샘플링 ---
fictitious_sales = df[df['Class'] == 1].copy()
fictitious_sales['TransactionType'] = 'Fictitious_Sale'
fictitious_sales['TransactionDate'] = pd.to_datetime(fictitious_sales['TransactionDate'])
fictitious_sales['Quarter'] = fictitious_sales['TransactionDate'].dt.quarter

print("\n[사기 수법 고도화] 불규칙/중첩 분기별 N:M 지급 시나리오를 적용합니다...")
all_kickbacks_list = []
for quarter in range(1, 5):
    quarterly_fraud = fictitious_sales[fictitious_sales['Quarter'] == quarter]
    payout_start_month = (quarter - 1) * 3 + 2
    payout_start_date = pd.to_datetime(f'2024-{payout_start_month:02d}-01')
    payout_days = 90
    fraud_summary = quarterly_fraud.groupby('Client_ID').agg(Total_Fraud_Amount=('Amount', 'sum')).reset_index()
    for _, row in fraud_summary.iterrows():
        total_kickback_amount = row['Total_Fraud_Amount'] * np.random.uniform(0.85, 0.95)
        num_splits = np.random.randint(2, 6)
        split_amounts = np.random.dirichlet(np.ones(num_splits)) * total_kickback_amount
        for i in range(num_splits):
            random_day = np.random.randint(1, payout_days + 1)
            kickback_date = payout_start_date + pd.to_timedelta(random_day, unit='d')
            all_kickbacks_list.append({'Date': kickback_date, 'Client_ID': row['Client_ID'], 'TransactionType': 'Kickback_Withdrawal', 'Amount': split_amounts[i]})
kickbacks = pd.DataFrame(all_kickbacks_list)
print(f"-> {len(fictitious_sales)}건의 가공매출에 대해 {len(kickbacks)}건의 분기별 분할 리베이트를 생성했습니다.")

print("\n[사기 수법 고도화] 리베이트 중 30%를 공범 계좌로 이전합니다...")
accomplice_indices = kickbacks.sample(frac=0.3, random_state=42).index
random_accomplice_ids = np.random.choice(normal_clients, size=len(accomplice_indices))
kickbacks.loc[accomplice_indices, 'Client_ID'] = random_accomplice_ids

monthly_expenses = []
for month in range(1, 13):
    monthly_expenses.extend([
        {'Date': pd.to_datetime(f'2024-{month:02d}-25 10:00:00'), 'Client_ID': 'Internal_HR', 'Transaction_Info': 'Normal', 'Original_Amount': np.random.uniform(150000, 200000), 'Transaction_Description': f'{month}월분 전 임직원 급여이체'},
        {'Date': pd.to_datetime(f'2024-{month:02d}-10 15:00:00'), 'Client_ID': 'Building_Mng', 'Transaction_Info': 'Normal', 'Original_Amount': 50000, 'Transaction_Description': f'{month}월분 본점 임대료'}
    ])
monthly_expenses_df = pd.DataFrame(monthly_expenses)

num_total_samples = 20000
num_normal_samples = num_total_samples - len(fictitious_sales) - len(kickbacks) - len(monthly_expenses_df)
normal_df = df[df['Class'] == 0]
normal_sample_df = normal_df.sample(n=num_normal_samples, random_state=42)
normal_sample_df['TransactionType'] = 'Normal'

bank_df_combined = pd.concat([fictitious_sales, kickbacks, normal_sample_df, monthly_expenses_df]).sort_values(by='Date')

# --- 5단계: 최종 통장 입출금 내역 생성 ---
bank_statement = pd.DataFrame()
bank_statement['Date'] = pd.to_datetime(bank_df_combined['Date'])
bank_statement['Client_ID'] = bank_df_combined['Client_ID']
bank_statement['Transaction_Info'] = bank_df_combined['TransactionType']
bank_statement['Transaction_Description'] = bank_df_combined.get('Transaction_Description')
bank_statement['Original_Amount'] = bank_df_combined['Amount'].fillna(bank_df_combined.get('Original_Amount')).round(2)
bank_statement['Deposit'] = np.where(bank_statement['Transaction_Info'] == 'Fictitious_Sale', bank_statement['Original_Amount'], 0)
bank_statement['Withdrawal'] = np.where(bank_statement['Transaction_Info'] == 'Kickback_Withdrawal', bank_statement['Original_Amount'], 0)
is_normal_random = (bank_statement['Transaction_Info'] == 'Normal') & (~bank_statement['Client_ID'].isin(['Internal_HR', 'Building_Mng']))
is_normal_deposit = is_normal_random & (np.random.rand(len(bank_statement)) < 0.2)
is_normal_withdrawal = is_normal_random & ~is_normal_deposit
bank_statement.loc[is_normal_deposit, 'Deposit'] = bank_statement.loc[is_normal_deposit, 'Original_Amount']
bank_statement.loc[is_normal_withdrawal, 'Withdrawal'] = bank_statement.loc[is_normal_withdrawal, 'Original_Amount']
bank_statement.loc[bank_statement['Client_ID'].isin(['Internal_HR', 'Building_Mng']), 'Withdrawal'] = bank_statement.loc[bank_statement['Client_ID'].isin(['Internal_HR', 'Building_Mng']), 'Original_Amount']

# --- 6단계: 거래 내용 구체화 ---
desc_dict = {'Normal_Deposit': ['입점매장 판매수수료', 'SSG.COM PG사 정산입금'], 'Normal_Withdrawal': ['입점브랜드 판매대금 지급', 'VMD(매장연출) 업체 용역비'], 'Fictitious_Sale': ['(주)신세계인터내셔날 PB상품 매출', '(주)프라임컨설팅 입점계약금'], 'Kickback_Withdrawal': ['신규사업타당성 컨설팅 비용', 'VIP 고객 트렌드 분석용역']}
needs_desc_mask = bank_statement['Transaction_Description'].isnull()
for index, row in bank_statement[needs_desc_mask].iterrows():
    month, client, info, deposit = row['Date'].month, row['Client_ID'], row['Transaction_Info'], row['Deposit']
    if info == 'Normal':
        desc_key = f"{info}_{'Deposit' if deposit > 0 else 'Withdrawal'}"
        desc = f"{np.random.choice(desc_dict[desc_key])} ({client if 'Withdrawal' in desc_key else str(month)+'월분'})"
    else:
        desc_key = info
        desc = f"{np.random.choice(desc_dict[desc_key])} ({client})"
    bank_statement.loc[index, 'Transaction_Description'] = desc

# --- 7단계: 최종 정리 및 특성 공학 ---
final_bank_statement = bank_statement[['Date', 'Client_ID', 'Transaction_Description', 'Transaction_Info', 'Deposit', 'Withdrawal']].copy()
print("\n[특성 공학] AI 모델을 위한 새로운 힌트(Feature)를 추가합니다...")
final_bank_statement = final_bank_statement.sort_values(by=['Client_ID', 'Date']).reset_index(drop=True)
final_bank_statement['Time_Delta_Seconds'] = final_bank_statement.groupby('Client_ID')['Date'].diff().dt.total_seconds().fillna(0)
is_weekend = final_bank_statement['Date'].dt.dayofweek >= 5
is_off_hours = (final_bank_statement['Date'].dt.hour < 9) | (final_bank_statement['Date'].dt.hour >= 18)
final_bank_statement['Is_OffHours'] = (is_weekend | is_off_hours).astype(int)
final_bank_statement['Total_Amount'] = final_bank_statement['Deposit'] + final_bank_statement['Withdrawal']
client_avg_amount = final_bank_statement.groupby('Client_ID')['Total_Amount'].transform('mean')
final_bank_statement['Amount_vs_Avg'] = final_bank_statement['Total_Amount'] / (client_avg_amount + 1e-6)
final_bank_statement = final_bank_statement.drop(columns=['Total_Amount'])
final_bank_statement = final_bank_statement.sort_values(by='Date').reset_index(drop=True)
final_bank_statement['Balance'] = 100000000 + final_bank_statement['Deposit'].cumsum() - final_bank_statement['Withdrawal'].cumsum()
print("특성 공학 완료!")

# --- 최종 파일 저장 [경로 수정 완료] ---
ml_dataset_filename = "data/shinsegae_bank_statement_For_ML.csv"
final_bank_statement.to_csv(ml_dataset_filename, index=False)
print(f"\n✅ [Part 2] 완료: {len(final_bank_statement):,}건의 학습용 데이터셋이 '{ml_dataset_filename}'으로 저장되었습니다.")
print("-" * 50)