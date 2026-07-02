import pandas as pd
from src.infrastructure.aws_aurora_dsql import AuroraDsqlClient



class ModelRegistry:
    
    def __init__(self):
        self.aurora_client = AuroraDsqlClient()


    def load_selected_model_performance(self) -> pd.DataFrame:
        sql_query = """
            select symbol, combo_id, avg_upper_threshold, avg_lower_threshold, prediction_range
            from training_output.selected_model_performance
        """
        
        rows = self.aurora_client.dsql_execute_sql(sql_query)
        return rows



    def load_selected_model_combos(self, training_version: str = "models") -> pd.DataFrame:
        if training_version == "models":
            sql_query = """
                SELECT B.* 
                FROM training_output.selected_models AS A
                INNER JOIN training_output.model_performance AS B
                    ON A.combo_id = B.combo_id
                    AND A.symbol = B.symbol
                ORDER BY A.symbol, A.combo_id;
            """
        else:
            sql_query = f"""
                SELECT B.* 
                FROM training_output.selected_models_{training_version} AS A
                INNER JOIN training_output.model_performance_{training_version} AS B
                    ON A.combo_id = B.combo_id
                    AND A.symbol = B.symbol
                ORDER BY A.symbol, A.combo_id;
            """
        
        model_combos = self.aurora_client.dsql_execute_sql(sql_query)
        return model_combos



    def load_all_model_performance(self, training_version: str) -> pd.DataFrame:
        sql_query = f"""
            select * from training_output.model_performance_{training_version}
        """

        rows = self.aurora_client.dsql_execute_sql(sql_query)
        return rows