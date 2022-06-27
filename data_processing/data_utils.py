import json
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm


def get_df_from_json(json_path:str, min_length:float=0.01)->pd.DataFrame:
    """
    Generates a Dataframe containing utterrances infos from json annotations

    Args:
        json_path (str): path to json file containing annotations
        min_length (float): minimum length in seconds to count a non-speech utterance as it is

    Returns:
        pd.DataFrame: Annotation in the form of Dataframe
    """
    data = json.load(open(json_path, 'r'))

    speech_df = pd.DataFrame(data['speech_segments'])
    speech_df['utt_time'] = ""
    speech_df['speech'] = 1
    speech_df['start_time'] = np.round(speech_df['start_time'].astype(float),3)
    speech_df['end_time'] = np.round(speech_df['end_time'].astype(float),3)

    utt_times = []
    # Add start and stops for non_speech:
    for i in range(len(speech_df)-1):
        if abs(speech_df.iloc[i]['end_time'] - speech_df.iloc[i+1]['start_time']) > min_length:
            speech_df = speech_df.append({'start_time':speech_df.iloc[i]['end_time'], 
                                          'end_time':speech_df.iloc[i+1]['start_time'],
                                          'speech':0}, ignore_index=True)

    # Add first row
    if speech_df.iloc[0].start_time != 0.0:
        speech_df = speech_df.append({'start_time':0.0, 
                                      'end_time':speech_df.iloc[0]['start_time'],
                                      'speech':0}, ignore_index=True)

    # Add utterances times
    utt_times = []
    for i in range(len(speech_df)):
        utt_times.append(np.round(speech_df.iloc[i]['end_time'] - speech_df.iloc[i]['start_time'], 3))
    speech_df['utt_time'] = utt_times

    # Sort DF
    speech_df = speech_df.sort_values(by='start_time').reset_index(drop=True)
    # Add audio ID
    speech_df['audio_id'] = json_path.replace('.json', '.wav')

    return speech_df


def get_consolidated_dataframe(json_data_path:str)->pd.DataFrame:
    """
    Get consolidated data for speech and non-speech utterances in the form of a Dataframe.
    - For each json annotation, get its corresponding Dataframe
    - Concatenate all Dataframes

    Args:
        json_data_path (str): Path to folder containing JSON annotations

    Returns:
        pd.DataFrame: Consolidated Dataframe
    """

    list_json = glob(json_data_path + '*.json')
    
    speech_dfs = []
    for k in tqdm(list_json):
        speech_df = get_df_from_json(k)
        speech_dfs.append(speech_df)

    speech_df = pd.concat(speech_dfs).reset_index(drop=True)
    
    return speech_df