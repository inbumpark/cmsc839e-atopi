# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Loads the SCIN dataset from the CSV files in Google Cloud Storage.

Setup (one time):
  pip install -r requirements.txt  (see README)

Authenticate (one time per machine):
  gcloud auth application-default login

  This sets up Application Default Credentials for the google-cloud-storage
  client. Requires the gcloud CLI (https://cloud.google.com/sdk/docs/install).
  In Colab, use instead:
    from google.colab import auth
    auth.authenticate_user()

Usage:
  import data_parser
  df = data_parser.load_dataset()

Export all Eczema cases to data/cases.jsonl + data/images/ (resumable):
  python data_parser.py [--limit N]
"""

import argparse
import ast
import io
import json
import os

# import matplotlib.pyplot as plt  # only for the commented-out display helpers
import pandas as pd
from google.cloud import storage
# from PIL import Image


class Globals:
  # GCP project with GCS bucket of interest
  gcp_project = 'dx-scin-public'

  # GCS bucket with data to read
  gcs_bucket_name = 'dx-scin-public-data'

  # CSV of case metadata to read
  cases_csv = 'dataset/scin_cases.csv'

  # CSV of label metadata to read
  labels_csv = 'dataset/scin_labels.csv'

  # Images directory
  gcs_images_dir = 'dataset/images/'

  ### Key column names
  image_path_columns = ['image_1_path', 'image_2_path', 'image_3_path']
  weighted_skin_condition_label = "weighted_skin_condition_label"
  skin_condition_label = "dermatologist_skin_condition_on_label_name"

  ###### Formed during execution:

  # Client for querying GCS
  gcs_storage_client = None

  # Bucket object for loading files
  gcs_bucket = None

  # pd.DataFrame for the loaded metadata_csv
  cases_df = None

  # pd.DataFrame for the loaded labels_csv
  cases_and_labels_df = None


# Notebook helper from scin_demo.ipynb, unused by the export; uncomment to debug bucket contents.
# def list_blobs(storage_client, bucket_name):
#   """Helper to list blobs in a bucket (useful for debugging)."""
#   blobs = storage_client.list_blobs(bucket_name)
#   for blob in blobs:
#     print(blob)

def initialize_df_with_metadata(bucket, csv_path):
  """Loads the given CSV into a pd.DataFrame."""
  df = pd.read_csv(io.BytesIO(bucket.blob(csv_path).download_as_string()), dtype={'case_id': str})
  df['case_id'] = df['case_id'].astype(str)
  return df

def augment_metadata_with_labels(df, bucket, csv_path):
  """Loads the given CSV into a pd.DataFrame."""
  labels_df = pd.read_csv(io.BytesIO(bucket.blob(csv_path).download_as_string()), dtype={'case_id': str})
  labels_df['case_id'] = labels_df['case_id'].astype(str)
  merged_df = pd.merge(df, labels_df, on='case_id')
  return merged_df

def load_dataset():
  """Connects to GCS, loads the cases and labels CSVs, and returns the merged pd.DataFrame."""
  Globals.gcs_storage_client = storage.Client(Globals.gcp_project)
  Globals.gcs_bucket = Globals.gcs_storage_client.bucket(
      Globals.gcs_bucket_name
  )
  Globals.cases_df = initialize_df_with_metadata(Globals.gcs_bucket, Globals.cases_csv)
  Globals.cases_and_labels_df = augment_metadata_with_labels(Globals.cases_df, Globals.gcs_bucket, Globals.labels_csv)
  return Globals.cases_and_labels_df

# Notebook display helpers from scin_demo.ipynb, unused by the export. To use them, uncomment
# them along with the matplotlib/PIL imports at the top.
# def display_image(bucket, image_path):
#   image = Image.open(io.BytesIO(bucket.blob(image_path).download_as_string()))
#   figure_size=4
#   f, axarr = plt.subplots(1, 1, figsize = (figure_size, figure_size))
#   axarr.imshow(image, cmap='gray')
#   axarr.axis('off')
#   plt.show()
#
# def display_images_for_case(df, case_id="", print_condition_labels=True):
#   # Use a random case if none is provided:
#   if case_id:
#     matched_df = df[df['case_id'] == case_id]
#   else:
#     matched_df = df.sample(1)
#
#   image_paths = matched_df[Globals.image_path_columns].values.tolist()[0]
#   for path in image_paths:
#     if isinstance(path, str):
#       display_image(Globals.gcs_bucket, path)
#   if print_condition_labels:
#     condition_labels = matched_df[[Globals.weighted_skin_condition_label]].values.tolist()[0]
#     for label in condition_labels:
#       if isinstance(label, str):
#         print(label)


### Eczema export: data/cases.jsonl + data/images/

DURATIONS = {
    'ONE_DAY': '1 day',
    'LESS_THAN_ONE_WEEK': 'less than 1 week',
    'ONE_TO_FOUR_WEEKS': '1 to 4 weeks',
    'ONE_TO_THREE_MONTHS': '1 to 3 months',
    'THREE_TO_TWELVE_MONTHS': '3 to 12 months',
    'MORE_THAN_ONE_YEAR': 'more than 1 year',
    'MORE_THAN_FIVE_YEARS': 'more than 5 years',
    'SINCE_CHILDHOOD': 'since childhood',
}

# Values that mean "not answered"; these fields are left out of the context.
UNKNOWN_VALUES = {'UNKNOWN', 'AGE_UNKNOWN', 'OTHER_OR_UNSPECIFIED', 'NONE_IDENTIFIED'}


def _parse(entry, default):
  """Parses a stringified Python list/dict cell, returning default for empty cells."""
  return ast.literal_eval(entry) if isinstance(entry, str) else default

def get_case_ids_with_condition(df, condition='Eczema', label_column=Globals.weighted_skin_condition_label):
  """Returns the case_ids whose condition label contains the given condition."""
  has_condition = df[label_column].apply(lambda entry: condition in _parse(entry, {}))
  return df.loc[has_condition, 'case_id'].tolist()

def download_image(bucket, gcs_path, local_path):
  """Downloads one image from GCS unless it is already on disk."""
  if not os.path.exists(local_path):
    bucket.blob(gcs_path).download_to_filename(local_path)

def _yes_columns(row, prefix):
  """Comma-joined names of the prefix* columns answered YES, e.g. 'arm, leg'."""
  names = [c[len(prefix):] for c in row.index if c.startswith(prefix) and row[c] == 'YES']
  return ', '.join('none' if n.startswith('no_relevant') else n.replace('_', ' ') for n in names)

def _known(value):
  return isinstance(value, str) and value not in UNKNOWN_VALUES

def build_context(row):
  """Self-reported patient context, keyed like context_fields in configs/default.yaml."""
  duration, age, sex, fst = (row[c] for c in ['condition_duration', 'age_group', 'sex_at_birth', 'fitzpatrick_skin_type'])
  context = {
      'duration': DURATIONS.get(duration, duration.lower().replace('_', ' ')) if _known(duration) else None,
      'body_location': _yes_columns(row, 'body_parts_'),
      'texture': _yes_columns(row, 'textures_'),
      'symptoms': _yes_columns(row, 'condition_symptoms_'),
      'other_symptoms': _yes_columns(row, 'other_symptoms_'),
      'age': age[len('AGE_'):].replace('_TO_', '-') if _known(age) else None,  # AGE_30_TO_39 -> 30-39
      'sex': sex.lower() if _known(sex) else None,
      'fitzpatrick': fst if _known(fst) else None,
  }
  return {k: v for k, v in context.items() if v}

def build_case_record(row, image_paths, label):
  """One cases.jsonl record for the case in row."""
  names = _parse(row['dermatologist_skin_condition_on_label_name'], [])
  confidences = _parse(row['dermatologist_skin_condition_confidence'], [])
  label_confidences = [c for n, c in zip(names, confidences) if n == label]
  return {
      'case_id': row['case_id'],
      'images': image_paths,
      'label': label,
      'label_confidence': max(label_confidences) if label_confidences else None,
      'differential': _parse(row[Globals.weighted_skin_condition_label], {}),
      'context': build_context(row),
  }

def _exported_case_ids(jsonl_path):
  """case_ids already written to jsonl_path (empty if the file doesn't exist yet)."""
  if not os.path.exists(jsonl_path):
    return set()
  with open(jsonl_path, encoding='utf-8') as f:
    return {json.loads(line)['case_id'] for line in f if line.strip()}

def export_cases(df, case_ids, condition='Eczema', out_dir='data'):
  """Downloads each case's images to out_dir/images/ and appends its record to out_dir/cases.jsonl.

  Cases already in cases.jsonl are skipped, so the export can be stopped and resumed.
  """
  images_dir = os.path.join(out_dir, 'images')
  os.makedirs(images_dir, exist_ok=True)
  jsonl_path = os.path.join(out_dir, 'cases.jsonl')
  done = _exported_case_ids(jsonl_path)
  todo = [c for c in case_ids if c not in done]
  print(f'{len(done)} cases already exported, {len(todo)} to go')
  rows = df.set_index('case_id', drop=False)

  written = 0
  with open(jsonl_path, 'a', encoding='utf-8') as f:
    for case_num, case_id in enumerate(todo, start=1):
      row = rows.loc[case_id]
      try:
        image_paths = []
        gcs_paths = [p for p in row[Globals.image_path_columns] if isinstance(p, str)]
        for n, gcs_path in enumerate(gcs_paths, start=1):
          name = f'{case_id}_{n}{os.path.splitext(gcs_path)[1]}'
          download_image(Globals.gcs_bucket, gcs_path, os.path.join(images_dir, name))
          image_paths.append(f'images/{name}')
      except Exception as e:
        print(f'  skipping case {case_id}: {e}')
        continue
      f.write(json.dumps(build_case_record(row, image_paths, condition)) + '\n')
      f.flush()
      written += 1
      if case_num % 25 == 0:
        print(f'  {case_num}/{len(todo)}')
  print(f'wrote {written} cases; {len(done) + written} total in {jsonl_path}')


if __name__ == '__main__':
  print(f'GCS bucket name: {Globals.gcs_bucket_name}')
  print(f'cases_csv: {Globals.cases_csv}')
  print(f'labels_csv: {Globals.labels_csv}')
  print(f'images dir: {Globals.gcs_images_dir}')
  parser = argparse.ArgumentParser()
  parser.add_argument('--limit', type=int, default=None, help='export at most N cases (for testing)')
  args = parser.parse_args()
  df = load_dataset()
  print(len(df))
  case_ids = get_case_ids_with_condition(df, 'Eczema')
  print(f'{len(case_ids)} Eczema cases')
  export_cases(df, case_ids[:args.limit], 'Eczema')
