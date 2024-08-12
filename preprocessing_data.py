import csv
import pandas as pd


if __name__ == '__main__':
    # 데이터 불러오기
    df = pd.read_csv("./data/HW_JJ_LDGS_CFR_RSTRNT_PREFEER_INFO_202303.csv")

    # 데이터 전처리 - 최신 데이터만 추출
    data_sorted = df.sort_values(by='BASE_YM', ascending=False)
    latest_data = data_sorted.drop_duplicates(subset=['LDGS_NM', 'RSTRNT_NM'], keep='first')
    grouped_data = latest_data.groupby('LDGS_NM')

    # 데이터 전처리 - training 형태로 변환
    output_data = []

    for name, group in grouped_data:
        lodgment = f"숙박업명: {name} - 주소: {group['LDGS_ADDR'].iloc[0]}"
        
        restaurants_list = []
        for _, row in group.iterrows():
            restaurant_info = (
                f"식당명: {row['RSTRNT_NM']} - 식당주소: {row['RSTRNT_ADDR']}, "
                f"Latitude: {row['RSTRNT_LA']}, Longitude: {row['RSTRNT_LO']}, "
                f"제주도민 매출금액 비율: {row['JJINHBT_SALES_PRICE_RATE']}, "
                f"외지인 매출금액 비율: {row['OTSD_SALES_PRICE_RATE']}, "
                f"전체 매출금액 비율: {row['ALL_SALES_PRICE_RATE']}"
            )
            restaurants_list.append(restaurant_info)
        
        restaurants = "\n".join(restaurants_list)
        
        output_data.append({
            "lodgment": f"""{lodgment}""",
            "nearby_restaurants": f"근처 식당 정보\n{restaurants}"
        })

    # 데이터 저장
    output_df = pd.DataFrame(output_data)
    latest_output_file_path = './data/jeju_preprocessed.csv'
    output_df.to_csv(latest_output_file_path, index=False)