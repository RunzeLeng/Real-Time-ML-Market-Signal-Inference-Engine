from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd



class ModelSignalService:


    def model_predict_with_group_average(
        self,
        X,
        models: dict,
        symbol: str,
        combo_id: int,
        lower_threshold: float,
        upper_threshold: float,
        prediction_range: str = "30m",
    ) -> str | None:
        
        matching_models = [
            model
            for model_name, model in models.items()
            if model_name.startswith(f"{symbol}_{prediction_range}_{combo_id}_")
        ]

        if not matching_models:
            return None

        sell_probs = [
            model.predict_proba(X)[0, 0]
            for model in matching_models
        ]

        avg_prob = sum(sell_probs) / len(sell_probs)

        if avg_prob >= upper_threshold:
            return "sell"
        if avg_prob <= lower_threshold:
            return "buy"

        return None



    def predict_symbol_combo_signals(
        self,
        X,
        models: dict,
        selected_model_df: pd.DataFrame,
        max_workers: int = 8,
    ) -> pd.DataFrame:
        
        def run_one_combo(row):
            symbol = row["symbol"]
            combo_id = row["combo_id"]
            lower_threshold = row["avg_lower_threshold"]
            upper_threshold = row["avg_upper_threshold"]
            prediction_range = row["prediction_range"]

            signal = self.model_predict_with_group_average(
                X=X,
                models=models,
                symbol=symbol,
                combo_id=combo_id,
                lower_threshold=lower_threshold,
                upper_threshold=upper_threshold,
                prediction_range=prediction_range,
            )

            return {
                "symbol": symbol,
                "combo_id": combo_id,
                "signal": signal,
            }

        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(run_one_combo, row)
                for _, row in selected_model_df.iterrows()
            ]

            for future in as_completed(futures):
                results.append(future.result())

        return pd.DataFrame(results)



    def symbol_voting_system(self, signal_df: pd.DataFrame, id_df: pd.DataFrame) -> pd.DataFrame:
        
        vote_map = {"sell": 1, "buy": -1, None: 0}
        working_df = signal_df.copy()
        working_df["vote"] = working_df["signal"].map(vote_map).fillna(0)

        symbol_votes = (
            working_df.groupby("symbol", as_index=False)["vote"]
            .sum()
            .rename(columns={"vote": "vote_sum"})
        )

        symbol_votes["final_signal"] = symbol_votes["vote_sum"].apply(
            lambda x: "sell" if x >= 3 else "buy" if x <= -3 else None
        )

        output_df = symbol_votes[["symbol", "final_signal"]].copy()
        output_df["id"] = id_df.iloc[0]["id"]

        return output_df[["id", "symbol", "final_signal"]]



    def merge_post_signal_and_validation_dfs(
        self,
        signal_prediction: pd.DataFrame,
        post_df: pd.DataFrame,
        validation_df: pd.DataFrame,
    ) -> pd.DataFrame:
        
        filtered_signal_prediction = signal_prediction[
            signal_prediction["final_signal"].notna()
        ].copy()

        prepared_post_df = post_df.drop(columns=["created_at"], errors="ignore").rename(
            columns={"created_at_seconds": "created_at"}
        )

        merged_df = prepared_post_df.merge(
            filtered_signal_prediction,
            on="id",
            how="inner",
        )

        merged_df = merged_df.merge(
            validation_df,
            left_on=["symbol", "final_signal"],
            right_on=["symbol", "predicted_signal"],
            how="inner",
        )
        
        merged_df["combined_score"] = (
            merged_df["reasonableness_score"] * merged_df["market_impact_score"]
        )
        
        return merged_df[
            [
            "id",
            "created_at",
            "content",
            "symbol",
            "predicted_signal",
            "market_impact_score",
            "reasonableness_score",
            "brief_reason",
            "combined_score",
            ]
        ]



    def score_decision_layer(
        self,
        df: pd.DataFrame,
        symbol_threshold: float = 0.7,
        combined_threshold: float = 0.4,
    ) -> pd.DataFrame:
        scored_df = df.copy()
        
        scored_df = scored_df[
            (scored_df["reasonableness_score"] >= symbol_threshold) &
            (scored_df["combined_score"] >= combined_threshold)
        ].copy()

        return scored_df



    def calculate_processing_latency(self, df: pd.DataFrame) -> pd.DataFrame:
        
        latency_df = df.copy()
        current_time = pd.Timestamp.now(tz="US/Eastern")
        created_at_time = pd.to_datetime(latency_df.iloc[0]["created_at"])

        latency_df["latency"] = round((current_time - created_at_time).total_seconds(), 2)
        
        return latency_df
