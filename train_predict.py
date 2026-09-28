import os
import sys
import time
import re
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

# Определение базовой папки скрипта для работы из любой директории
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from metric import precision_at_recall, recall_at_fpr

# Фиксация случайного сида для полной воспроизводимости
SEED = 42
np.random.seed(SEED)

print("=" * 60)
print("ДЕТЕКЦИЯ БОТОВ НА АВИТО - ОБУЧЕНИЕ И ГЕНЕРАЦИЯ САБМИТА")
print("=" * 60)

# 1. Загрузка данных
print("\n[1/5] Загрузка датасетов...")
t0 = time.time()
train_path = os.path.join(BASE_DIR, 'data', 'train.csv')
test_path = os.path.join(BASE_DIR, 'data', 'test.csv')
events_path = os.path.join(BASE_DIR, 'data', 'events.csv.gz')

train = pd.read_csv(train_path, parse_dates=['cookie_created_at', 'window_start_ts', 'window_end_ts'])
test = pd.read_csv(test_path, parse_dates=['cookie_created_at', 'window_start_ts', 'window_end_ts'])
events = pd.read_csv(events_path, parse_dates=['event_ts'])

print(f"Загружено: train {train.shape}, test {test.shape}, events {events.shape} за {time.time() - t0:.1f}с")

# 2. Строгая фильтрация событий строго внутри окна наблюдения (предотвращение утечки данных)
print("\n[2/5] Фильтрация событий строго внутри окна наблюдения [window_start_ts, window_end_ts)...")
def get_in_window(ev, meta):
    m = ev.merge(meta[['cookie_id', 'window_start_ts', 'window_end_ts']], on='cookie_id')
    return m[(m.event_ts >= m.window_start_ts) & (m.event_ts < m.window_end_ts)].copy()

ev_tr = get_in_window(events, train)
ev_te = get_in_window(events, test)
print(f"Событий внутри окна: train {len(ev_tr)} (отброшено {len(events.merge(train[['cookie_id']], on='cookie_id')) - len(ev_tr)} событий из будущего), test {len(ev_te)}")

# 3. Извлечение доменных поведенческих признаков
print("\n[3/5] Генерация доменных признаков поведения...")
def extract_features(meta_df, ev_df):
    start_t = time.time()
    feats = {}
    meta_indexed = meta_df.set_index('cookie_id')
    cookie_ids = meta_df['cookie_id'].values
    
    # 3.1 Метаданные куки
    cookie_age_s = (meta_indexed['window_start_ts'] - meta_indexed['cookie_created_at']).dt.total_seconds()
    feats['cookie_age_hours'] = (cookie_age_s / 3600.0).values
    feats['cookie_age_days'] = (cookie_age_s / 86400.0).values
    feats['cookie_age_log'] = np.log1p(np.maximum(0, cookie_age_s / 3600.0)).values
    feats['cookie_is_new_1d'] = (cookie_age_s <= 86400.0).astype(int).values
    feats['cookie_is_new_7d'] = (cookie_age_s <= 7 * 86400.0).astype(int).values
    feats['cookie_is_new_30d'] = (cookie_age_s <= 30 * 86400.0).astype(int).values
    
    w_start = meta_indexed['window_start_ts']
    feats['window_dayofweek'] = w_start.dt.dayofweek.values
    feats['window_is_weekend'] = w_start.dt.dayofweek.isin([5, 6]).astype(int).values
    
    # Сортировка событий по куке и времени
    ev = ev_df.sort_values(['cookie_id', 'event_ts']).copy()
    ev['platform_clean'] = ev['platform'].astype(str).str.lower()
    
    # 3.2 Разбор юзер-агента и сигнатуры парсеров
    ua_str = ev['user_agent'].astype(str).str.lower()
    ev['ua_is_headless'] = ua_str.str.contains('headless', regex=False).astype(int)
    ev['ua_is_python'] = ua_str.str.contains('python|urllib', regex=True).astype(int)
    ev['ua_is_scrapy'] = ua_str.str.contains('scrapy', regex=False).astype(int)
    ev['ua_is_curl'] = ua_str.str.contains('curl', regex=False).astype(int)
    ev['ua_is_okhttp'] = ua_str.str.contains('okhttp', regex=False).astype(int)
    ev['ua_is_linux_desktop'] = (ua_str.str.contains('linux', regex=False) & ~ua_str.str.contains('android', regex=False)).astype(int)
    ev['ua_is_windows'] = ua_str.str.contains('windows', regex=False).astype(int)
    ev['ua_is_mac'] = ua_str.str.contains('macintosh|mac os', regex=True).astype(int)
    ev['ua_is_android_ua'] = ua_str.str.contains('android', regex=False).astype(int)
    ev['ua_is_ios_ua'] = ua_str.str.contains('iphone|ipad|ios', regex=True).astype(int)
    ev['ua_is_chrome'] = (ua_str.str.contains('chrome', regex=False) & ~ua_str.str.contains('headless', regex=False)).astype(int)
    ev['ua_is_firefox'] = ua_str.str.contains('firefox', regex=False).astype(int)
    ev['ua_is_yabrowser'] = ua_str.str.contains('yabrowser', regex=False).astype(int)
    ev['ua_is_explicit_scraper'] = (ev['ua_is_headless'] | ev['ua_is_python'] | ev['ua_is_scrapy'] | ev['ua_is_curl']).astype(int)
    
    # Извлечение мажорной версии браузера или приложения
    def extract_version(s):
        match = re.search(r'(?:chrome|firefox|yabrowser|avito|version)/(\d+)', s)
        return int(match.group(1)) if match else 0
    ev['browser_ver'] = ua_str.apply(extract_version)
    
    # 3.3 Общие объемы событий и тайминги
    g = ev.groupby('cookie_id')
    ev_counts = g.size().reindex(cookie_ids, fill_value=0)
    feats['n_events'] = ev_counts.values
    feats['n_events_log'] = np.log1p(feats['n_events'])
    
    first_ts = g['event_ts'].min().reindex(cookie_ids)
    last_ts = g['event_ts'].max().reindex(cookie_ids)
    w_start_series = meta_indexed['window_start_ts']
    w_end_series = meta_indexed['window_end_ts']
    
    feats['time_to_first_event_h'] = (first_ts - w_start_series).dt.total_seconds().div(3600).fillna(24.0).values
    feats['time_from_last_event_h'] = (w_end_series - last_ts).dt.total_seconds().div(3600).fillna(24.0).values
    dur_s = (last_ts - first_ts).dt.total_seconds().fillna(0)
    feats['active_duration_s'] = dur_s.values
    feats['active_duration_hours'] = (dur_s / 3600.0).values
    feats['active_duration_ratio'] = (dur_s / 86400.0).values
    feats['event_rate_per_min'] = (feats['n_events'] / (dur_s / 60.0 + 1.0)).values
    
    # 3.4 Межсобытийные интервалы (статистики дельта t)
    ev['dt'] = g['event_ts'].diff().dt.total_seconds()
    dt_valid = ev.dropna(subset=['dt'])
    dt_g = dt_valid.groupby('cookie_id')['dt']
    
    feats['dt_mean'] = dt_g.mean().reindex(cookie_ids, fill_value=0).values
    feats['dt_std'] = dt_g.std().fillna(0).reindex(cookie_ids, fill_value=0).values
    feats['dt_median'] = dt_g.median().reindex(cookie_ids, fill_value=0).values
    feats['dt_min'] = dt_g.min().reindex(cookie_ids, fill_value=0).values
    feats['dt_max'] = dt_g.max().reindex(cookie_ids, fill_value=0).values
    feats['dt_q25'] = dt_g.quantile(0.25).reindex(cookie_ids, fill_value=0).values
    feats['dt_q75'] = dt_g.quantile(0.75).reindex(cookie_ids, fill_value=0).values
    feats['dt_iqr'] = feats['dt_q75'] - feats['dt_q25']
    feats['dt_cv'] = feats['dt_std'] / (feats['dt_mean'] + 1e-5)
    
    # Параметр всплесковости B = (std - mean) / (std + mean)
    denom = feats['dt_std'] + feats['dt_mean']
    feats['burstiness'] = np.where(denom > 0, (feats['dt_std'] - feats['dt_mean']) / np.maximum(denom, 1e-5), 0.0)
    
    # Доли быстрых действий и длительных пауз
    for thresh in [0.5, 1.0, 2.0, 5.0, 10.0, 60.0, 300.0]:
        t_col = f'dt_lt_{str(thresh).replace(".", "_")}s'
        ev[t_col] = (ev['dt'] < thresh).astype(int)
        cnt = ev.groupby('cookie_id')[t_col].sum().reindex(cookie_ids, fill_value=0).values
        feats[f'{t_col}_count'] = cnt
        feats[f'{t_col}_ratio'] = cnt / np.maximum(1, feats['n_events'])
        
    ev['dt_gt_60s'] = (ev['dt'] > 60.0).astype(int)
    ev['dt_gt_300s'] = (ev['dt'] > 300.0).astype(int)
    feats['dt_gt_60s_ratio'] = ev.groupby('cookie_id')['dt_gt_60s'].sum().reindex(cookie_ids, fill_value=0).values / np.maximum(1, feats['n_events'])
    feats['dt_gt_300s_ratio'] = ev.groupby('cookie_id')['dt_gt_300s'].sum().reindex(cookie_ids, fill_value=0).values / np.maximum(1, feats['n_events'])
    
    # Выделение сессий (пауза между событиями более 30 минут)
    ev['is_new_session'] = (ev['dt'] > 1800.0).astype(int)
    sess_cnt = (ev.groupby('cookie_id')['is_new_session'].sum() + 1).reindex(cookie_ids, fill_value=1).values
    feats['n_sessions'] = sess_cnt
    feats['events_per_session'] = feats['n_events'] / sess_cnt
    
    # 3.5 Суточная активность и энтропия по часам
    ev['hour'] = ev['event_ts'].dt.hour
    feats['n_active_hours'] = g['hour'].nunique().reindex(cookie_ids, fill_value=0).values
    night_cnt = ev[ev['hour'].between(1, 6)].groupby('cookie_id').size().reindex(cookie_ids, fill_value=0).values
    feats['night_events_count'] = night_cnt
    feats['night_events_ratio'] = night_cnt / np.maximum(1, feats['n_events'])
    
    # Расчет энтропии распределения по часам суток
    def calc_entropy(s):
        probs = s.value_counts(normalize=True).values
        return -np.sum(probs * np.log2(probs + 1e-9))
    feats['hour_entropy'] = g['hour'].apply(calc_entropy).reindex(cookie_ids, fill_value=0).values
    
    hour_counts = ev.groupby(['cookie_id', 'hour']).size()
    max_in_hour = hour_counts.groupby('cookie_id').max().reindex(cookie_ids, fill_value=0).values
    feats['max_events_in_hour'] = max_in_hour
    feats['max_hourly_event_share'] = max_in_hour / np.maximum(1, feats['n_events'])
    
    # 3.6 Типы событий, объемы и конверсии
    ev_types = ['item_view', 'search_results_view', 'photo_swipe', 'favorite_add', 
                'seller_page_view', 'contact_phone_show', 'login', 'contact_chat_open', 'contact_message_sent']
    for et in ev_types:
        col_cnt = ev[ev['event_name'] == et].groupby('cookie_id').size().reindex(cookie_ids, fill_value=0).values
        feats[f'cnt_{et}'] = col_cnt
        feats[f'ratio_{et}'] = col_cnt / np.maximum(1, feats['n_events'])
        
    feats['cnt_contacts_all'] = feats['cnt_contact_phone_show'] + feats['cnt_contact_chat_open'] + feats['cnt_contact_message_sent']
    feats['ratio_contacts_all'] = feats['cnt_contacts_all'] / np.maximum(1, feats['n_events'])
    feats['contacts_per_item'] = feats['cnt_contacts_all'] / (feats['cnt_item_view'] + 1.0)
    feats['photo_per_item'] = feats['cnt_photo_swipe'] / (feats['cnt_item_view'] + 1.0)
    feats['fav_per_item'] = feats['cnt_favorite_add'] / (feats['cnt_item_view'] + 1.0)
    feats['seller_per_item'] = feats['cnt_seller_page_view'] / (feats['cnt_item_view'] + 1.0)
    feats['item_per_search'] = feats['cnt_item_view'] / (feats['cnt_search_results_view'] + 1.0)
    
    feats['has_login'] = (feats['cnt_login'] > 0).astype(int)
    feats['has_contact'] = (feats['cnt_contacts_all'] > 0).astype(int)
    feats['has_favorite'] = (feats['cnt_favorite_add'] > 0).astype(int)
    feats['has_photo_swipe'] = (feats['cnt_photo_swipe'] > 0).astype(int)
    feats['item_views_without_search'] = ((feats['cnt_item_view'] > 0) & (feats['cnt_search_results_view'] == 0)).astype(int)
    
    # Марковские переходы (цепочки последовательных событий)
    ev['prev_event_name'] = g['event_name'].shift(1)
    ev['trans_item_item'] = ((ev['prev_event_name'] == 'item_view') & (ev['event_name'] == 'item_view')).astype(int)
    ev['trans_search_item'] = ((ev['prev_event_name'] == 'search_results_view') & (ev['event_name'] == 'item_view')).astype(int)
    ev['trans_search_search'] = ((ev['prev_event_name'] == 'search_results_view') & (ev['event_name'] == 'search_results_view')).astype(int)
    ev['trans_item_photo'] = ((ev['prev_event_name'] == 'item_view') & (ev['event_name'] == 'photo_swipe')).astype(int)
    
    feats['trans_item_item_cnt'] = ev.groupby('cookie_id')['trans_item_item'].sum().reindex(cookie_ids, fill_value=0).values
    feats['trans_search_item_cnt'] = ev.groupby('cookie_id')['trans_search_item'].sum().reindex(cookie_ids, fill_value=0).values
    feats['trans_search_search_cnt'] = ev.groupby('cookie_id')['trans_search_search'].sum().reindex(cookie_ids, fill_value=0).values
    feats['trans_item_photo_cnt'] = ev.groupby('cookie_id')['trans_item_photo'].sum().reindex(cookie_ids, fill_value=0).values
    
    feats['trans_item_item_ratio'] = feats['trans_item_item_cnt'] / np.maximum(1, feats['n_events'])
    feats['trans_search_item_ratio'] = feats['trans_search_item_cnt'] / np.maximum(1, feats['n_events'])
    feats['trans_search_search_ratio'] = feats['trans_search_search_cnt'] / np.maximum(1, feats['n_events'])
    feats['trans_item_photo_ratio'] = feats['trans_item_photo_cnt'] / np.maximum(1, feats['n_events'])
    
    feats['event_type_entropy'] = g['event_name'].apply(calc_entropy).reindex(cookie_ids, fill_value=0).values
    
    # Серии подряд идущих одинаковых действий
    ev['is_item'] = (ev['event_name'] == 'item_view').astype(int)
    ev['is_search'] = (ev['event_name'] == 'search_results_view').astype(int)
    ev['item_block'] = (ev['is_item'] == 0).cumsum()
    ev['search_block'] = (ev['is_search'] == 0).cumsum()
    
    item_streaks = ev[ev['is_item'] == 1].groupby(['cookie_id', 'item_block']).size()
    max_item_s = item_streaks.groupby('cookie_id').max().reindex(cookie_ids, fill_value=0).values
    feats['max_item_streak'] = max_item_s
    feats['max_item_streak_ratio'] = max_item_s / np.maximum(1, feats['cnt_item_view'])
    
    search_streaks = ev[ev['is_search'] == 1].groupby(['cookie_id', 'search_block']).size()
    max_search_s = search_streaks.groupby('cookie_id').max().reindex(cookie_ids, fill_value=0).values
    feats['max_search_streak'] = max_search_s
    feats['max_search_streak_ratio'] = max_search_s / np.maximum(1, feats['cnt_search_results_view'])
    
    # 3.7 Разнообразие объявлений, категорий, локаций и типы продавцов
    item_nu = g['item_id'].nunique().reindex(cookie_ids, fill_value=0).values
    feats['item_nunique'] = item_nu
    feats['item_repeat_ratio'] = (feats['cnt_item_view'] - item_nu) / (feats['cnt_item_view'] + 1.0)
    
    cat_nu = g['item_category'].nunique().reindex(cookie_ids, fill_value=0).values
    feats['cat_nunique'] = cat_nu
    loc_nu = g['item_location'].nunique().reindex(cookie_ids, fill_value=0).values
    feats['loc_nunique'] = loc_nu
    feats['loc_per_item'] = loc_nu / (item_nu + 1.0)
    feats['cat_per_item'] = cat_nu / (item_nu + 1.0)
    feats['items_per_cat'] = item_nu / (cat_nu + 1.0)
    feats['items_per_loc'] = item_nu / (loc_nu + 1.0)
    
    # Продавцы (профессиональные и частные)
    ev['is_pro'] = (ev['seller_type'] == 'pro').astype(int)
    ev['is_private'] = (ev['seller_type'] == 'private').astype(int)
    cnt_pro = ev.groupby('cookie_id')['is_pro'].sum().reindex(cookie_ids, fill_value=0).values
    cnt_priv = ev.groupby('cookie_id')['is_private'].sum().reindex(cookie_ids, fill_value=0).values
    feats['cnt_pro'] = cnt_pro
    feats['cnt_private'] = cnt_priv
    feats['ratio_pro'] = cnt_pro / (cnt_pro + cnt_priv + 1e-5)
    
    # 3.8 Поисковые запросы и глубина выдачи
    feats['search_query_nunique'] = g['search_query'].nunique().reindex(cookie_ids, fill_value=0).values
    sq_notna = ev['search_query'].dropna().astype(str)
    ev['query_len'] = sq_notna.str.len()
    feats['query_len_mean'] = ev.groupby('cookie_id')['query_len'].mean().reindex(cookie_ids, fill_value=0).values
    feats['query_len_max'] = ev.groupby('cookie_id')['query_len'].max().reindex(cookie_ids, fill_value=0).values
    
    feats['search_page_max'] = g['search_page'].max().reindex(cookie_ids, fill_value=0).values
    feats['search_page_mean'] = g['search_page'].mean().reindex(cookie_ids, fill_value=0).values
    feats['search_page_std'] = g['search_page'].std().reindex(cookie_ids, fill_value=0).fillna(0).values
    feats['has_deep_search'] = (feats['search_page_max'] >= 5).astype(int)
    feats['has_very_deep_search'] = (feats['search_page_max'] >= 10).astype(int)
    
    # 3.9 Кинематика мыши (координаты pointer_x, pointer_y)
    ev['has_pointer'] = ev['pointer_x'].notna().astype(int)
    ptr_cnt = ev.groupby('cookie_id')['has_pointer'].sum().reindex(cookie_ids, fill_value=0).values
    feats['pointer_count'] = ptr_cnt
    feats['pointer_ratio'] = ptr_cnt / np.maximum(1, feats['n_events'])
    feats['has_any_pointer'] = (ptr_cnt > 0).astype(int)
    
    ptr_ev = ev.dropna(subset=['pointer_x', 'pointer_y']).copy()
    if len(ptr_ev) > 0:
        ptr_g = ptr_ev.groupby('cookie_id')
        feats['ptr_x_mean'] = ptr_g['pointer_x'].mean().reindex(cookie_ids, fill_value=0).values
        feats['ptr_x_std'] = ptr_g['pointer_x'].std().fillna(0).reindex(cookie_ids, fill_value=0).values
        feats['ptr_y_mean'] = ptr_g['pointer_y'].mean().reindex(cookie_ids, fill_value=0).values
        feats['ptr_y_std'] = ptr_g['pointer_y'].std().fillna(0).reindex(cookie_ids, fill_value=0).values
        
        ptr_ev['dx'] = ptr_g['pointer_x'].diff()
        ptr_ev['dy'] = ptr_g['pointer_y'].diff()
        ptr_ev['dist'] = np.sqrt(ptr_ev['dx']**2 + ptr_ev['dy']**2)
        ptr_ev['dt_ptr'] = ptr_g['event_ts'].diff().dt.total_seconds()
        ptr_ev['speed'] = ptr_ev['dist'] / (ptr_ev['dt_ptr'] + 1e-4)
        
        dist_g = ptr_ev.dropna(subset=['dist']).groupby('cookie_id')
        feats['ptr_dist_sum'] = dist_g['dist'].sum().reindex(cookie_ids, fill_value=0).values
        feats['ptr_dist_mean'] = dist_g['dist'].mean().reindex(cookie_ids, fill_value=0).values
        feats['ptr_dist_max'] = dist_g['dist'].max().reindex(cookie_ids, fill_value=0).values
        feats['ptr_speed_mean'] = dist_g['speed'].mean().reindex(cookie_ids, fill_value=0).values
        feats['ptr_speed_max'] = dist_g['speed'].max().reindex(cookie_ids, fill_value=0).values
        
        ptr_ev['xy'] = ptr_ev['pointer_x'].astype(str) + '_' + ptr_ev['pointer_y'].astype(str)
        xy_nu = ptr_ev.groupby('cookie_id')['xy'].nunique().reindex(cookie_ids, fill_value=0).values
        feats['ptr_xy_nunique'] = xy_nu
        feats['ptr_repeat_ratio'] = 1.0 - (xy_nu / (ptr_cnt + 1e-5))
    
    # 3.10 Платформы и активность без движения мыши
    feats['platform_nunique'] = g['platform_clean'].nunique().reindex(cookie_ids, fill_value=0).values
    for plat in ['android', 'ios', 'iphone', 'desktop', 'web']:
        p_cnt = ev[ev['platform_clean'] == plat].groupby('cookie_id').size().reindex(cookie_ids, fill_value=0).values
        feats[f'plat_cnt_{plat}'] = p_cnt
        feats[f'plat_ratio_{plat}'] = p_cnt / np.maximum(1, feats['n_events'])
        
    is_desktop_or_web = (feats['plat_cnt_desktop'] > 0) | (feats['plat_cnt_web'] > 0)
    feats['desktop_zero_pointer'] = (is_desktop_or_web & (feats['pointer_count'] == 0)).astype(int)
    
    # 3.11 Агрегаты юзер-агентов
    feats['ua_nunique'] = g['user_agent'].nunique().reindex(cookie_ids, fill_value=0).values
    feats['browser_ver_max'] = g['browser_ver'].max().reindex(cookie_ids, fill_value=0).values
    feats['browser_ver_min'] = g['browser_ver'].min().reindex(cookie_ids, fill_value=0).values
    
    for ua_feat in ['ua_is_headless', 'ua_is_python', 'ua_is_scrapy', 'ua_is_curl', 'ua_is_okhttp',
                    'ua_is_linux_desktop', 'ua_is_windows', 'ua_is_mac', 'ua_is_android_ua', 'ua_is_ios_ua',
                    'ua_is_chrome', 'ua_is_firefox', 'ua_is_yabrowser', 'ua_is_explicit_scraper']:
        feats[f'{ua_feat}_any'] = ev.groupby('cookie_id')[ua_feat].max().reindex(cookie_ids, fill_value=0).values
        feats[f'{ua_feat}_ratio'] = ev.groupby('cookie_id')[ua_feat].mean().reindex(cookie_ids, fill_value=0).values
        
    feats['mismatch_mobile_plat_desktop_ua'] = ((feats['plat_cnt_android'] > 0) & (feats['ua_is_windows_any'] == 1)).astype(int)
    feats['mismatch_desktop_plat_mobile_ua'] = ((feats['plat_cnt_desktop'] > 0) & (feats['ua_is_android_ua_any'] == 1)).astype(int)
    
    df_result = pd.DataFrame(feats, index=cookie_ids)
    df_result.index.name = 'cookie_id'
    df_result = df_result.reset_index()
    print(f"Сгенерировано {len(df_result.columns)-1} признаков для {len(meta_df)} строк за {time.time() - start_t:.1f}с")
    return df_result

Xtr = extract_features(train, ev_tr)
Xte = extract_features(test, ev_te)

feature_cols = [c for c in Xtr.columns if c not in ['cookie_id', 'target']]
ytr = train['target'].values
print(f"\nРазмерность финального пространства признаков: {len(feature_cols)} колонок")

# 4. Проверка качества на временном валидационном сплите (Out-of-Time: Val >= 2026-04-16)
print("\n[4/5] Валидация на отложенном временном окне (Val >= 2026-04-16)...")
val_mask = train.window_start_ts.ge('2026-04-16').values
X_tr_val, y_tr_val = Xtr.loc[~val_mask, feature_cols], ytr[~val_mask]
X_va_val, y_va_val = Xtr.loc[val_mask, feature_cols], ytr[val_mask]

# Обучение моделей на валидационном сплите
m_lgb_val = lgb.LGBMClassifier(n_estimators=650, learning_rate=0.025, num_leaves=31, min_child_samples=25, subsample=0.8, colsample_bytree=0.75, random_state=SEED, verbose=-1)
m_lgb_val.fit(X_tr_val, y_tr_val)
p_val_lgb = m_lgb_val.predict_proba(X_va_val)[:, 1]

m_cb_val = CatBoostClassifier(iterations=850, learning_rate=0.03, depth=6, l2_leaf_reg=4, random_seed=SEED, verbose=0)
m_cb_val.fit(X_tr_val, y_tr_val)
p_val_cb = m_cb_val.predict_proba(X_va_val)[:, 1]

m_xgb_val = xgb.XGBClassifier(n_estimators=550, learning_rate=0.025, max_depth=5, subsample=0.8, colsample_bytree=0.75, random_state=SEED, eval_metric='logloss')
m_xgb_val.fit(X_tr_val, y_tr_val)
p_val_xgb = m_xgb_val.predict_proba(X_va_val)[:, 1]

p_val_ens = 0.20 * p_val_cb + 0.35 * p_val_lgb + 0.45 * p_val_xgb

print(f"Результаты валидации:")
print(f"  LightGBM P@R>=0.70: {precision_at_recall(y_va_val, p_val_lgb):.4f} | Recall@1%FPR: {recall_at_fpr(y_va_val, p_val_lgb):.4f}")
print(f"  CatBoost P@R>=0.70: {precision_at_recall(y_va_val, p_val_cb):.4f} | Recall@1%FPR: {recall_at_fpr(y_va_val, p_val_cb):.4f}")
print(f"  XGBoost  P@R>=0.70: {precision_at_recall(y_va_val, p_val_xgb):.4f} | Recall@1%FPR: {recall_at_fpr(y_va_val, p_val_xgb):.4f}")
print(f"  Ансамбль P@R>=0.70: {precision_at_recall(y_va_val, p_val_ens):.4f} | Recall@1%FPR: {recall_at_fpr(y_va_val, p_val_ens):.4f}")

# 5. Обучение мультисидного ансамбля на 100% обучающих данных и инференс
print("\n[5/5] Обучение мультисидного ансамбля на 100% train данных...")
X_full = Xtr[feature_cols]
y_full = ytr
X_test = Xte[feature_cols]

test_preds = np.zeros(len(test), dtype=np.float64)
seeds = [42, 2026, 777]

for s in seeds:
    print(f"  Обучение с сидом {s}...")
    # Модель LightGBM
    lgb_full = lgb.LGBMClassifier(n_estimators=750, learning_rate=0.025, num_leaves=31, min_child_samples=25, subsample=0.8, colsample_bytree=0.75, random_state=s, verbose=-1)
    lgb_full.fit(X_full, y_full)
    p_lgb = lgb_full.predict_proba(X_test)[:, 1]
    
    # Модель CatBoost
    cb_full = CatBoostClassifier(iterations=950, learning_rate=0.03, depth=6, l2_leaf_reg=4, random_seed=s, verbose=0)
    cb_full.fit(X_full, y_full)
    p_cb = cb_full.predict_proba(X_test)[:, 1]
    
    # Модель XGBoost
    xgb_full = xgb.XGBClassifier(n_estimators=650, learning_rate=0.025, max_depth=5, subsample=0.8, colsample_bytree=0.75, random_state=s, eval_metric='logloss')
    xgb_full.fit(X_full, y_full)
    p_xgb = xgb_full.predict_proba(X_test)[:, 1]
    
    seed_pred = 0.20 * p_cb + 0.35 * p_lgb + 0.45 * p_xgb
    test_preds += seed_pred / len(seeds)

# Формирование итогового файла сабмита
sub = pd.DataFrame({
    'cookie_id': test['cookie_id'],
    'score': test_preds
})

# Валидация формата и корректности сабмита
print("\nПроверка корректности сабмита:")
print(f"  Размерность: {sub.shape} (Ожидалось: ({len(test)}, 2))")
assert len(sub) == len(test), "Не совпадает число строк с test.csv!"
assert list(sub.columns) == ['cookie_id', 'score'], "Неверные имена колонок!"
assert sub['score'].isna().sum() == 0, "Обнаружены NaN значения!"
assert sub['score'].between(0, 1).all(), "Значения score выходят за границы [0, 1]!"
assert (sub['cookie_id'] == test['cookie_id']).all(), "Порядок cookie_id нарушен!"

# Сохранение в submission.csv
sub_path = os.path.join(BASE_DIR, 'submission.csv')
sub.to_csv(sub_path, index=False)
print(f"Файл {sub_path} успешно сохранен!")
print(sub.head(10))
print("\nСтатистика распределения предсказанных вероятностей:")
print(sub['score'].describe())
