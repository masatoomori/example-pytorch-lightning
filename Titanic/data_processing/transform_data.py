import os
import json
import datetime

import pandas as pd
import numpy as np

HOME_PATH = os.pardir

INPUT_PATH = os.path.join(HOME_PATH, 'input', 'original')
OUTPUT_PATH = os.path.join(HOME_PATH, 'input', 'preprocessed')
os.makedirs(OUTPUT_PATH, exist_ok=True)

MODELING_DATA_FILE = 'modeling.{}'
SUBMISSION_DATA_FILE = 'submission.{}'
DATA_PROFILE = 'data_profile.json'

TARGET_COL = 'Survived'


def load_data():
    df_modeling = pd.read_csv(os.path.join(INPUT_PATH, MODELING_DATA_FILE.format('csv')), encoding='utf8', dtype=object)
    df_submission = pd.read_csv(os.path.join(INPUT_PATH, SUBMISSION_DATA_FILE.format('csv')), encoding='utf8', dtype=object)
    for df in [df_modeling, df_submission]:
        for c in ['SibSp', 'Parch', 'Survived']:
            if c in df.columns:
                df[c] = df[c].astype(int)

    for df in [df_modeling, df_submission]:
        for c in ['Fare']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c])

    df_both = pd.concat([df_modeling, df_submission], sort=False).reset_index(drop=True)

    return df_both, df_modeling, df_submission


def cabin(df):
    # Cabinにいたか
    df['in_cabin'] = 0
    df.loc[df['Cabin'].notnull(), 'in_cabin'] = 0

    # Cabinを複数人でシェアしていたか
    df_c = df[['Cabin']].copy()
    df_c['cabin_shared'] = 1
    df_c = df_c.groupby('Cabin', as_index=False).sum()

    df = pd.merge(df, df_c, how='left', on=['Cabin'])
    df['cabin_shared'].fillna(0, inplace=True)
    df['cabin_shared'] = df['cabin_shared'].apply(lambda x: 2 if x > 1 else 0)

    return df


def family(df):
    # 家族数
    df['familySize'] = df['SibSp'] + df['Parch'] + 1
    df['is_alone'] = df['familySize'].apply(lambda x: 1 if x == 1 else 0)

    # 家族属性
    df['is_large_family'] = 0
    df.loc[df['familySize'] > 4, 'is_large_family'] = 1

    return df


def passenger_name(df):
    # 敬称を抽出
    df['Salutation'] = 'others'
    df.loc[df['Name'].apply(lambda x: ', Miss.' in x), 'Salutation'] = 'Miss'
    df.loc[df['Name'].apply(lambda x: ', Mlle.' in x), 'Salutation'] = 'Miss'
    df.loc[df['Name'].apply(lambda x: ', Ms.' in x), 'Salutation'] = 'Miss'
    df.loc[df['Name'].apply(lambda x: ', Mr.' in x), 'Salutation'] = 'Mr'
    df.loc[df['Name'].apply(lambda x: ', Mrs.' in x), 'Salutation'] = 'Mrs'
    df.loc[df['Name'].apply(lambda x: ', Mme' in x), 'Salutation'] = 'Mrs'
    df.loc[df['Name'].apply(lambda x: ', Sir' in x), 'Salutation'] = 'Sir'
    df.loc[df['Name'].apply(lambda x: ', Master' in x), 'Salutation'] = 'Master'
    df.loc[df['Name'].apply(lambda x: ', Rev.' in x), 'Salutation'] = 'Rev'
    df.loc[df['Name'].apply(lambda x: ', Don.' in x), 'Salutation'] = 'Rev'
    # df.loc[df['Name'].apply(lambda x: ', Dr.' in x), 'Salutation'] = 'Dr'

    # 敬称を平均生存率への影響度に置き換える
    salutations = list(set(df['Salutation']))
    ave_survival_ratio = df['Survived'].mean()

    salutation_impact = dict()
    for s in salutations:
        salutation_impact.update({s: df[df['Salutation'] != s]['Survived'].mean() - ave_survival_ratio})
    df['salutation_impact'] = df['Salutation'].apply(lambda x: salutation_impact[x])

    # 家族の中の少年かどうか
    df['is_family_boy'] = 0
    df.loc[(df['Salutation'] == 'Master') & (df['familySize'].between(2, 4)), 'is_family_boy'] = 1

    return df


def ticket(df):
    # Golden Ticket
    df['golden_ticket'] = 0
    df.loc[df['Ticket'].apply(lambda x: len(x) == 4), 'golden_ticket'] = 1
    df.loc[df['Ticket'].apply(lambda x: len(x) == 5 and x[0] in ('1', '2')), 'golden_ticket'] = 1
    df.loc[df['Ticket'].apply(lambda x: len(x) == 6 and x[0] == '3'), 'golden_ticket'] = 1
    df.loc[df['Ticket'].apply(lambda x: len(x) == 7 and x.startswith('PP')), 'golden_ticket'] = 1
    df.loc[df['Ticket'].apply(lambda x: len(x) == 7 and x.startswith('C ')), 'golden_ticket'] = 1
    df.loc[df['Ticket'].apply(lambda x: len(x) == 9 and x.startswith('C.')), 'golden_ticket'] = 1
    df.loc[df['Ticket'].apply(lambda x: len(x) == 10 and x.startswith('C.')), 'golden_ticket'] = 1
    df.loc[df['Ticket'].apply(lambda x: x.startswith('PC ')), 'golden_ticket'] = 1
    df.loc[df['Ticket'].apply(lambda x: x.startswith('STON/')), 'golden_ticket'] = 1

    # チケット価格
    df_fare_med = df[['Pclass', 'familySize', 'Fare']].groupby(['Pclass', 'familySize'], as_index=False).median()
    df_fare_med.rename(columns={'Fare': 'fare_med'}, inplace=True)
    df = pd.merge(df, df_fare_med, how='left', on=['Pclass', 'familySize'])
    df['high_fare'] = df.apply(lambda row: row['Fare'] / row['fare_med'], axis=1)

    return df


def embarked(df):
    # Embarked
    df['Embarked_n'] = 0
    df.loc[df['Embarked'] == 'Q', 'Embarked_n'] = 1
    df.loc[df['Embarked'] == 'C', 'Embarked_n'] = 2

    return df


def format_data(df):
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].astype(int)

    for c in ['Pclass']:
        df[c] = df[c].astype(int)

    for c in ['Age']:
        df[c] = df[c].astype(float)

    for c in ['Cabin']:
        df[c] = df[c].astype(str)

    return df


def fill_missing_values(df_target, df_base, drop_cols):
    # 欠測値をcul_colごとの数値の場合は中央値、文字列の場合は最頻値で埋める
    cut_col = 'Salutation'
    for c in df_target.columns:
        if c != TARGET_COL and c not in drop_cols:
            dtype = type(df_base[c].tolist()[0])
            print(c, dtype)
            if dtype is str:
                df_f = df_base.groupby(cut_col)[c].apply(lambda x: x.mode()).reset_index()[[cut_col, c]]
                df_f.rename(columns={c: 'na_value'}, inplace=True)
                df_target = pd.merge(df_target, df_f, how='left', on=cut_col)
                df_target[c].fillna(df_target['na_value'], inplace=True)
                df_target.drop('na_value', axis=1, inplace=True)
            elif dtype in (float, int):
                df_f = df_base[[cut_col, c]].groupby(cut_col).median().reset_index()
                df_f.rename(columns={c: 'na_value'}, inplace=True)
                df_target = pd.merge(df_target, df_f, how='left', on=cut_col)
                df_target[c].fillna(df_target['na_value'], inplace=True)
                df_target.drop('na_value', axis=1, inplace=True)

    return df_target


def save_data_profile(df_train, prediction_type):
    explanatory_cols = [c for c in df_train.columns if c != TARGET_COL]
    arr_explanatory_col = np.array(explanatory_cols, dtype=str)
    explanatory_dtype = dict(zip(explanatory_cols,
                                 [str(type(df_train[c].tolist()[0])) for c in explanatory_cols]))

    if prediction_type == 'classification':
        prof = {
            'created': datetime.datetime.now().isoformat(),
            'script':  __file__,
            'num_records': len(df_train),
            'prediction_type': prediction_type,
            'target': {
                'name': TARGET_COL,
                'dtype': str(type(df_train[TARGET_COL].tolist()[0])),
                'num_classes': len(set(df_train[TARGET_COL])),
                'classes': list(set(df_train[TARGET_COL]))
            },
            'explanatory': {
                'names': explanatory_cols,
                'dims': arr_explanatory_col.shape,
                'dtype': explanatory_dtype
            }
        }
    elif prediction_type == 'regression':
        prof = {
            'created': datetime.datetime.now().isoformat(),
            'script': __file__,
            'num_records': len(df_train),
            'prediction_type': prediction_type,
            'target': {
                'name': TARGET_COL,
                'dtype': str(type(df_train[TARGET_COL].tolist()[0])),
                'num_classes': 1,
                'classes': [min(df_train[TARGET_COL]), max(df_train[TARGET_COL])]
            },
            'explanatory': {
                'names': explanatory_cols,
                'dims': arr_explanatory_col.shape,
                'dtype': explanatory_dtype
            }
        }
    else:
        print('prediction type should be either regression or classification')
        prof = {}

    with open(os.path.join(OUTPUT_PATH, DATA_PROFILE), 'w') as f:
        json.dump(prof, f, indent=4)

    return prof


def main():
    drop_cols = ['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Salutation']

    df_both, df_modeling, df_submission = load_data()

    # 変数を作成する
    df_both = cabin(df_both)
    df_both = family(df_both)
    df_both = passenger_name(df_both)
    df_both = ticket(df_both)
    df_both = embarked(df_both)

    df_modeling = df_both[df_both['PassengerId'].isin(df_modeling['PassengerId'])].copy()
    df_submission = df_both[df_both['PassengerId'].isin(df_submission['PassengerId'])].copy()

    df_submission.drop(TARGET_COL, axis=1, inplace=True)

    df_modeling = format_data(df_modeling)
    df_submission = format_data(df_submission)

    df_submission = fill_missing_values(df_submission, df_modeling, drop_cols)  # df_modelingの内容が変わる前に行う
    df_modeling = fill_missing_values(df_modeling, df_modeling, drop_cols)

    df_modeling.drop(drop_cols, axis=1, inplace=True)
    df_submission.drop(drop_cols, axis=1, inplace=True)

    # modeling側で元データと突合できるよう、インデックスを整えておく
    df_modeling.reset_index(drop=True, inplace=True)
    df_submission.reset_index(drop=True, inplace=True)

    df_modeling.to_csv(os.path.join(OUTPUT_PATH, MODELING_DATA_FILE.format('csv')), encoding='utf8', index=False)
    df_submission.to_csv(os.path.join(OUTPUT_PATH, SUBMISSION_DATA_FILE.format('csv')), encoding='utf8', index=False)
    df_modeling.to_pickle(os.path.join(OUTPUT_PATH, MODELING_DATA_FILE.format('pkl')))
    df_submission.to_pickle(os.path.join(OUTPUT_PATH, SUBMISSION_DATA_FILE.format('pkl')))

    save_data_profile(df_modeling, 'classification')


def test():
    main()


if __name__ == '__main__':
    test()
