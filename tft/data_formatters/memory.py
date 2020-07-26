# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Custom formatting functions for Memory dataset.

Defines dataset specific column definitions and data transformations.
"""

import data_formatters.base
import libs.utils as utils
import sklearn.preprocessing

GenericDataFormatter = data_formatters.base.GenericDataFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes


class MemoryFormatter(GenericDataFormatter):
  """Defines and formats data for the Memory dataset.

  Attributes:
    column_definition: Defines input and data type of column used in the
      experiment.
    identifiers: Entity identifiers used in experiments.
  """

  _column_definition = [
      ('SysID', DataTypes.CATEGORICAL, InputTypes.ID),
      ('date', DataTypes.DATE, InputTypes.TIME),
      ('Mem_avg', DataTypes.REAL_VALUED, InputTypes.TARGET),
      ('ActiveTsEntries_sum', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('PlObj_sum', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('TsEntries_sum', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('ActiveTsEntries_max', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('PlObj_max', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('TsEntries_max', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('PlArea_N', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

      ('ODATA_TRANSFER_JOB_20', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/AIF/PERS_RUN_AUTO_REPROCESS', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/ALERT_ANA_FILL_BUFFER', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/ALERT_CREATE_NOTIF', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/BGPROCESS', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

      ('/IBP/CREATE_TIME_PERIODS', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/DISAGG_CLEAN_UP', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/FCPSRV_NO_SCM_EXECUTE', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/IBPLOG_ARCHIVE', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/INVALID_SESSION_CLEANUP', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

      ('/IBP/MIGRPA_CHECK_READINESS', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/MIGRPA_EXECUTE', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/MIGRPA_WATCHDOG_BGR', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/MODBO_ACTIVATE', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/MODBO_ACTIVATION_BGRND', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

      ('/IBP/PLAN_CALENDAR_CLEANUP', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/POST_INSTALLATION', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/RDATA_INT_POST_PROC', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/RHCI_DI', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/RMEA_CLEAN_UP', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

      ('/IBP/RM_ANALYTIC_OUTBOUND', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/RM_ASC_RULE_UPDATE', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/RM_ASC_SEGMENT_UPDATE', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/RM_DATA_MONITOR', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/RM_DELETE_EXPIRED_SIM', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

      ('/IBP/RM_DELETE_MD', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/RM_DMI_CVC_KF_GENERATION', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/RM_DMI_CVC_REGENERATE', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/RM_GARBAGE_COLLECT_OUT_CP', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/RM_GATING_FACTOR_ANALYSIS', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

      ('/IBP/RM_IBPF_PLEV_ADJUST', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/RM_INTEGRATE_DATA', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/RM_INT_CVC_CLEANUP', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/RM_INT_PRF_ACTIVATE', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/RM_NEW_PLANNING_RUN', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

      ('/IBP/RM_PLANNING_RUN', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/RM_PLANNING_RUN_OPTR', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/RM_SDI_LOAD_CONFIG', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/RM_UPDATE_VERSIONS', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/RM_VERSION_COPY_V2', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

      ('/IBP/RM_VERSION_DELETE_V2', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/ROP_COPY', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/ROP_COPYVS', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/ROP_COPY_TP', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/ROP_DDR', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

      ('/IBP/ROP_DELVS', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/ROP_DISAGG', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/ROP_DIS_DEL_FIXING', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/ROP_FCST', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/ROP_FCST_AUT', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

      ('/IBP/ROP_GEN_RT', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/ROP_IO', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/ROP_KPIPROFILE', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/ROP_SCM', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/ROP_SCM_EXPL', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

      ('/IBP/ROP_SCM_FC', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/ROP_SEGMENTATION', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/ROP_SNPSHT', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/ROP_SNPSHT_LAGBASED', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/ROP_SNPSHT_REDO', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

      ('/IBP/RSRV_DELETE_COMMENTS', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/RSRV_RLG_EXEC', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/R_AUTO_PROCESS_MGMT_STEP', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/R_DAEMON_WATCHDOG', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/R_JOB_OUTLIER_DETECTION', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

      ('/IBP/R_LOG_DELETE', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/R_PROCESS_MGMT_AUTOMATION', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/R_PROCESS_MGMT_AUTO_TECH', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/R_PROCESS_MGMT_DAILY_JOB', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/R_PURGE_CH', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

      ('/IBP/R_PURGE_DI', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/R_PURGE_DI_TEC', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/R_PURGE_KF', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/R_PURGE_KF_OUTSIDE_HORIZ', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/R_PURGE_MD', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

      ('/IBP/R_PURGE_NC', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/R_PURGE_OPENCURSOR_TEC', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/R_PURGE_PDCL_TEC', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/R_PURGE_PLANNING_AREA', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/R_PURGE_PROCESS_MGMT', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

      ('/IBP/R_VF_GEN', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/R_VF_REG_AFTER_IMPORT', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/R_VF_USER_ANALYSIS', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/TRIGGER_OUTBOUND_MESSAGE', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IBP/WBP_SESSION_CLEANUP', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

      ('/IBP/XLSAD_DAC_RECONCILE_CMNTS', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IWBEP/R_CLEAN_UP_QRL', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IWFND/R_SM_CLEANUP', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IWNGW/R_CLEAR_LOGS', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IWNGW/R_HUB_CLEAR_DATA4USER', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

      ('/IWXBE/R_EVENT_STATISTICS', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/IWXBE/R_SM_CLEANUP', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/UI5/UPD_ODATA_METADATA_CACHE', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/UIF/CHECK_LOAD_4_CONS_BG', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('/UIF/CLEAN_CUSTOMER_APPSTATES', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

      ('APS_COM_CA_EXEC_IDOC', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('ARBFND_FETCH_CXML_MSG', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('ATO_REFRESH_CHECK', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('ATO_REFRESH_READY_FOR_IMPORT', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('ATO_REPAIR_OR_REBUILD', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

      ('ATO_SYNCH_CUSTOMER_DOWNTIME', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('BUPTDTRANSMIT', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('CLMS_CRP_COLLECT_AND_UPLOAD', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('CLMS_HC_EXECUTE_CHECKS', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('DTINF_COLLECTION_MONITOR', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

      ('ESH_INT_REMSEARCH_REFRESH', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('ESH_JR_QUERY_LOG_REORG', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('ESH_QM_CALC_Q', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('ESH_QM_OPT', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('ODQ_TQ_WATCHDOG_JR', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

      ('ODQ_TQ_WORKER', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('RDAAG_AGING_RUN', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('REV_SYNC_FT_CONTENT', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('REV_SYNC_FT_CONTENT_TEST', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('RILM_DESTRUCTION_SCHEDULE', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

      ('RSARC_AOBJ_ARCHIVE_SCHEDULE_T', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('RSBCS_REORG', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('RSBPCOLL', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('RSBTCDELAPJ', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('RSBTCNOT', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

      ('RSCONN01', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('RSDBAJOB', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('RSDRAFT_LIFECYCLE_MANAGER', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('RSLDAGDS', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('RSN3_AGGR_REORG', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

      ('RSN3_STAT_COLLECTOR', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('RSPOPQ_ITEM_DEL', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('RSPOPQ_SPOOL2PQ', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('RSWEQSRV', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('RSWF_DATABASE_UPGRADE', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

      ('RSWF_OUTPUT_MANAGEMENT', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('RSWF_PUSH_NOTIFICATION_EXECUTE', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('RSWF_SYSTEM_ACTIONS', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('RSWF_SYSTEM_CLEANUP', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('RSWF_SYSTEM_DELAYED', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

      ('RSWF_SYSTEM_SCHEDULER', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('RSWF_SYSTEM_TEMPORARY', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('RSWF_UTL_EXECUTE_ACTIONS', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('RSWWCLEAR', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('RSWWCOND', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

      ('RSWWDHEX', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('RSWWRUNCNT', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('RSWWWIM', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('RSXMB_ARCHIVE_MESSAGES', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('RSXMB_ARCHIVE_PLAN', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

      ('RSXMB_DELETE_HISTORY', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('RSXMB_DELETE_MESSAGES', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('RSXMB_TABLE_SWITCH', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('RTM_COLLECT_ALL', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('R_JR_BTCJOBS_GENERATOR', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

      ('R_SILENT_DATA_MIGR_SCHEDULER', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('R_SILENT_DATA_MIGR_SUM', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('SADL_GW_CPL_LOG_CLEANUP', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('SBAL_DELETE', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('STC_TM_PROCESSOR', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

      ('SWNCTOTALT', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('SXCO_TRC_DELETE_TRACES', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('TMS_BCI_START_SERVICE', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('ZLA_NEW_NORMALIZATION', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

      # ('day_of_week', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      # ('day_of_month', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      # ('week_of_year', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      # ('month', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      # ('Region', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
  ]

  def __init__(self):
    """Initialises formatter."""

    self.identifiers = None
    self._real_scalers = None
    self._cat_scalers = None
    self._target_scaler = None
    self._num_classes_per_cat_input = None

  def split_data(self, df, valid_boundary=2016, test_boundary=2018):
    """Splits data frame into training-validation-test data frames.

    This also calibrates scaling object, and transforms data for each split.

    Args:
      df: Source data frame to split.
      valid_boundary: Starting year for validation data
      test_boundary: Starting year for test data

    Returns:
      Tuple of transformed (train, valid, test) data.
    """

    print('Formatting train-valid-test splits.')

    index = df['year']
    train = df.loc[index < valid_boundary]
    valid = df.loc[(index >= valid_boundary) & (index < test_boundary)]
    test = df.loc[index >= test_boundary]

    self.set_scalers(train)

    return (self.transform_inputs(data) for data in [train, valid, test])

  def set_scalers(self, df):
    """Calibrates scalers using the data supplied.

    Args:
      df: Data to use to calibrate scalers.
    """
    print('Setting scalers with training data...')

    column_definitions = self.get_column_definition()
    id_column = utils.get_single_col_by_input_type(InputTypes.ID,
                                                   column_definitions)
    target_column = utils.get_single_col_by_input_type(InputTypes.TARGET,
                                                       column_definitions)

    # Extract identifiers in case required
    self.identifiers = list(df[id_column].unique())

    # Format real scalers
    real_inputs = utils.extract_cols_from_data_type(
        DataTypes.REAL_VALUED, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    data = df[real_inputs].values
    self._real_scalers = sklearn.preprocessing.StandardScaler().fit(data)
    self._target_scaler = sklearn.preprocessing.StandardScaler().fit(
        df[[target_column]].values)  # used for predictions

    # Format categorical scalers
    categorical_inputs = utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    categorical_scalers = {}
    num_classes = []
    for col in categorical_inputs:
      # Set all to str so that we don't have mixed integer/string columns
      srs = df[col].apply(str)
      categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(
          srs.values)
      num_classes.append(srs.nunique())

    # Set categorical scaler outputs
    self._cat_scalers = categorical_scalers
    self._num_classes_per_cat_input = num_classes

  def transform_inputs(self, df):
    """Performs feature transformations.

    This includes both feature engineering, preprocessing and normalisation.

    Args:
      df: Data frame to transform.

    Returns:
      Transformed data frame.

    """
    output = df.copy()

    if self._real_scalers is None and self._cat_scalers is None:
      raise ValueError('Scalers have not been set!')

    column_definitions = self.get_column_definition()

    real_inputs = utils.extract_cols_from_data_type(
        DataTypes.REAL_VALUED, column_definitions,
        {InputTypes.ID, InputTypes.TIME})
    categorical_inputs = utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    # Format real inputs
    output[real_inputs] = self._real_scalers.transform(df[real_inputs].values)

    # Format categorical inputs
    for col in categorical_inputs:
      string_df = df[col].apply(str)
      output[col] = self._cat_scalers[col].transform(string_df)

    return output

  def format_predictions(self, predictions):
    """Reverts any normalisation to give predictions in original scale.

    Args:
      predictions: Dataframe of model predictions.

    Returns:
      Data frame of unnormalised predictions.
    """
    output = predictions.copy()

    column_names = predictions.columns

    for col in column_names:
      if col not in {'forecast_time', 'identifier'}:
        output[col] = self._target_scaler.inverse_transform(predictions[col])

    return output

  # Default params
  def get_fixed_params(self):
    """Returns fixed model parameters for experiments."""

    fixed_params = {
        'total_time_steps': 252 + 5,
        'num_encoder_steps': 252,
        'num_epochs': 100,
        'early_stopping_patience': 5,
        'multiprocessing_workers': 5,
    }

    return fixed_params

  def get_default_model_params(self):
    """Returns default optimised model parameters."""

    model_params = {
        'dropout_rate': 0.3,
        'hidden_layer_size': 160,
        'learning_rate': 0.01,
        'minibatch_size': 64,
        'max_gradient_norm': 0.01,
        'num_heads': 1,
        'stack_size': 1
    }

    return model_params
