import os

import pandas as pd

ORIGINAL_DATA_PATH = os.path.join('input', 'original')
PREPROCESSED_DATA_PATH = os.path.join('input', 'preprocessed')

TRAIN_ORIG_FILE = 'train.csv'
TEST_ORIG_FILE = 'test.csv'
MODELING_DATA_FILE = 'modeling.csv'
SUBMISSION_DATA_FILE = 'submission.csv'


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


def main():
    df_both, df_train, df_test = data_load()

    # 変数を作成する
    df_both = cabin(df_both)
    df_both = family(df_both)
    df_both = passenger_name(df_both)
    df_both = ticket(df_both)
    df_both = embarked(df_both)

    df_train = df_both[df_both['PassengerId'].isin(df_train['PassengerId'])]
    df_test = df_both[df_both['PassengerId'].isin(df_test['PassengerId'])]

    os.makedirs(PREPROCESSED_DATA_PATH, exist_ok=True)
    df_train.to_csv(os.path.join(PREPROCESSED_DATA_PATH, MODELING_DATA_FILE), encoding='utf8', index=False)
    df_test.to_csv(os.path.join(PREPROCESSED_DATA_PATH, SUBMISSION_DATA_FILE), encoding='utf8', index=False)


def test():
    main()


if __name__ == '__main__':
    test()
