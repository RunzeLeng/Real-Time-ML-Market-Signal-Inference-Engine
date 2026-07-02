import json
import pandas as pd



class LLMPromptBuilder:
    
    
    def build_validator_user_prompt(
        self,
        post_df: pd.DataFrame,
        signal_prediction: pd.DataFrame,
    ) -> str:
        
        post_content = str(post_df.iloc[0]["content"]).strip()
        filtered_predictions = signal_prediction[signal_prediction["final_signal"].notna()].copy()

        predicted_signals = filtered_predictions[["symbol", "final_signal"]].rename(
            columns={"final_signal": "predicted_signal"}
        ).to_dict(orient="records")

        predicted_signals_json = json.dumps(predicted_signals, ensure_ascii=False, indent=2)

        user_prompt = f"""
            Evaluate the following post and predicted ETF signals.

            Post content:
            {post_content}

            Predicted ETF signals:
            {predicted_signals_json}

            Please return valid JSON only.
        """

        return user_prompt