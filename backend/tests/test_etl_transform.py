"""ETL transform: CSVs with and without the Usage_Limit column.

tools/generate_mock_data.py (and older Kiro exports) do not emit Usage_Limit; the
transform must still run and fall back to Overage_Cap for usage_rate.
"""
import io

from backend.etl import transform as t

HEADER = ("Date,UserId,Client_Type,Chat_Conversations,Credits_Used,Overage_Cap,"
          "Overage_Credits_Used,Overage_Enabled,ProfileId,Subscription_Tier,Total_Messages,"
          "New_User,User_Email,auto_messages")


def _frame(rows, extra_header="", extra=None):
    lines = [HEADER + extra_header]
    for i, r in enumerate(rows):
        lines.append(r + ("," + extra[i] if extra else ""))
    return t._read_csv(io.StringIO("\n".join(lines) + "\n"))


ROW = "2026-10-01,u1,KIRO_IDE,3,50,200,0,false,p1,PRO,10,false,User@Example.com,10"


def test_missing_usage_limit_falls_back_to_overage_cap():
    df = _frame([ROW])
    assert "Usage_Limit" in df.columns            # filled in, not a KeyError
    fud, fmd = t.transform([df], {}, 0.04)
    r = fud.iloc[0]
    assert r["email"] == "user@example.com"
    assert r["usage_limit"] == 0
    assert abs(r["usage_rate"] - 50 / 200) < 1e-9   # Overage_Cap fallback
    assert not r["is_overage"]
    assert len(fmd) == 1


def test_present_usage_limit_is_used():
    df = _frame([ROW], extra_header=",Usage_Limit", extra=["1000"])
    fud, _ = t.transform([df], {}, 0.04)
    r = fud.iloc[0]
    assert r["usage_limit"] == 1000
    assert abs(r["usage_rate"] - 50 / 1000) < 1e-9
