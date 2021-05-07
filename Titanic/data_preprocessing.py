import os
import json
import datetime

import pandas as pd
import numpy as np

ORIGINAL_DATA_PATH = os.path.join('input', 'original')
PROCESSED_DATA_PATH = os.path.join('input', 'preprocessed')

TRAIN_ORIG_FILE = 'train.csv'
TEST_ORIG_FILE = 'test.csv'
MODELING_DATA_FILE = 'modeling.{}'
SUBMISSION_DATA_FILE = 'submission.{}'
DATA_PROFILE = 'data_profile.json'

TARGET_COL = 'Survived'


def data_load():
    df_train = pd.read_csv(os.path.join(ORIGINAL_DATA_PATH, TRAIN_ORIG_FILE), encoding='utf8', dtype=object)
    df_test = pd.read_csv(os.path.join(ORIGINAL_DATA_PATH, TEST_ORIG_FILE), encoding='utf8', dtype=object)
    for df in [df_train, df_test]:
        for c in ['SibSp', 'Parch', 'Survived']:
            if c in df.columns:
                df[c] = df[c].astype(int)

    for df in [df_train, df_test]:
        for c in ['Fare']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c])

    return df_train.append(df_test, sort=False), df_train, df_test


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

    return df


def save_data_profile(df_train):
    explanatory_cols = [c for c in df_train.columns if c != TARGET_COL]
    arr_explanatory_col = np.array(explanatory_cols, dtype=str)
    explanatory_dtype = dict(zip(explanatory_cols,
                                 [str(type(df_train[c].tolist()[0])) for c in explanatory_cols]))

    prof = {
        'created': datetime.datetime.now().isoformat(),
        'script':  __file__,
        'num_records': len(df_train),
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

    with open(os.path.join(PROCESSED_DATA_PATH, DATA_PROFILE), 'w') as f:
        json.dump(prof, f, indent=4)

    return prof


def main():
    df_both, df_train, df_test = data_load()

    # 変数を作成する
    df_both = cabin(df_both)
    df_both = family(df_both)
    df_both = passenger_name(df_both)
    df_both = ticket(df_both)
    df_both = embarked(df_both)

    df_train = df_both[df_both['PassengerId'].isin(df_train['PassengerId'])].copy()
    df_test = df_both[df_both['PassengerId'].isin(df_test['PassengerId'])].copy()

    drop_cols = ['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Salutation']
    df_train.drop(drop_cols, axis=1, inplace=True)
    df_test.drop(drop_cols, axis=1, inplace=True)
    df_test.drop(TARGET_COL, axis=1, inplace=True)

    df_train = format_data(df_train)
    df_test = format_data(df_test)

    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    df_train.to_csv(os.path.join(PROCESSED_DATA_PATH, MODELING_DATA_FILE.format('csv')), encoding='utf8', index=False)
    df_test.to_csv(os.path.join(PROCESSED_DATA_PATH, SUBMISSION_DATA_FILE.format('csv')), encoding='utf8', index=False)
    df_train.to_pickle(os.path.join(PROCESSED_DATA_PATH, MODELING_DATA_FILE.format('pkl')))
    df_test.to_pickle(os.path.join(PROCESSED_DATA_PATH, SUBMISSION_DATA_FILE.format('pkl')))

    save_data_profile(df_train)


def test():
    main()


if __name__ == '__main__':
    test()
