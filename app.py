import numpy as np
import pandas as pd
import streamlit as st
import portable

URL = "https://raw.githubusercontent.com/aaubs/ds-master/main/assignments/study-office/data/"

st.set_page_config(page_title="Study office: who to talk to", layout="wide")


@st.cache_resource
def load_model():
    return portable.Model("model")


@st.cache_data
def load_data():
    return pd.read_csv(URL + "history_week6.csv"), pd.read_csv(URL + "new_week6.csv")


model = load_model()
history, new = load_data()

# 2025 students: the model never trained on them, so the mistakes are honest
val = history[history["cohort"] == 2025].copy()
val["risk"] = model.predict_proba(val)
new["risk"] = model.predict_proba(new)

# ---------- sidebar: the costs (our own addition) ----------
st.sidebar.header("Costs (DKK)")
COST_TALK = st.sidebar.number_input("One conversation (adviser time)", value=500, step=100)
COST_WORRY = st.sidebar.number_input("One false alarm (a worried student)", value=2000, step=500)
COST_LEAVE = st.sidebar.number_input("One student who leaves", value=60000, step=5000)
HELPS = st.sidebar.slider("Share of at-risk students a conversation keeps", 0.0, 1.0, 0.30, 0.05)


def net_value(df, contacted):
    reached = (contacted & (df["left"] == 1)).sum()
    worried = (contacted & (df["left"] == 0)).sum()
    return reached * HELPS * COST_LEAVE - (reached + worried) * COST_TALK - worried * COST_WORRY


cuts = np.round(np.arange(0.02, 0.91, 0.01), 2)
values = [net_value(val, val["risk"] >= c) for c in cuts]
best_cut = float(cuts[int(np.argmax(values))])

# ---------- the rule ----------
st.title("Who should the study office talk to in week 6?")
rule = st.radio("The rule is based on", ["Number of conversations", "Risk cut-off"], horizontal=True)
if rule == "Number of conversations":
    n = st.slider("Conversations the office can hold", 0, 200, 40)
    contact_val = val["risk"].rank(ascending=False, method="first") <= n
    contact_new = new["risk"].rank(ascending=False, method="first") <= n
else:
    cut = st.slider("Contact students with a risk of at least", 0.02, 0.90, best_cut, 0.01)
    contact_val = val["risk"] >= cut
    contact_new = new["risk"] >= cut

tab1, tab2, tab3, tab4 = st.tabs(["This week's list", "The mistakes of this rule", "Per group", "What the costs say"])

# ---------- 1. this week's list ----------
with tab1:
    st.write(f"**{int(contact_new.sum())}** of the {len(new)} students in the 2026 list are marked for a conversation. "
             "A high risk means the student looks like students who left in earlier years. It is a reason to talk, not a verdict.")
    show = new.assign(**{"Talk to?": np.where(contact_new, "Yes", ""), "Risk of leaving": (new["risk"] * 100).round(0)})
    show = show.sort_values("risk", ascending=False)
    cols = ["Talk to?", "student_id", "Risk of leaving", "programme", "international", "fees_owed",
            "submitted_share", "missed_last3", "logins_last3", "weeks_since_login"]
    st.dataframe(show[cols], hide_index=True, width="stretch")
    st.download_button("Download the list (CSV)", show[cols].to_csv(index=False), "week6_list.csv")

# ---------- 2. the mistakes ----------
with tab2:
    left = val["left"] == 1
    reached = int((contact_val & left).sum())
    worried = int((contact_val & ~left).sum())
    missed = int((~contact_val & left).sum())
    st.subheader("What this rule would have done to the 2025 students")
    c1, c2, c3 = st.columns(3)
    c1.metric("Reached in time", reached)
    c2.metric("Worried for nothing", worried)
    c3.metric("Missed", missed)
    contacted = reached + worried
    if contacted > 0:
        st.write(f"Of the **{contacted}** students contacted, **{reached / contacted:.0%}** were really at risk (precision).")
    st.write(f"Of the **{reached + missed}** students who left, **{reached / max(1, reached + missed):.0%}** were reached (recall). "
             f"The other **{missed}** left without anyone trying to help.")
    st.write(f"Net value of this rule with the costs in the sidebar: **{net_value(val, contact_val):,.0f} DKK**.")

# ---------- 3. per group ----------
with tab3:
    st.subheader("The same numbers for international and domestic students (2025)")
    rows = []
    for name, flag in [("Domestic", 0), ("International", 1)]:
        g = val["international"] == flag
        l = val["left"] == 1
        r = int((g & contact_val & l).sum())
        w = int((g & contact_val & ~l).sum())
        m = int((g & ~contact_val & l).sum())
        rows.append({"Group": name, "Students": int(g.sum()), "Really left": r + m, "Contacted": r + w,
                     "Reached in time": r, "Worried for nothing": w, "Missed": m,
                     "Share of leavers reached": f"{r / max(1, r + m):.0%}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Few students leave in each group, so one student more or less moves these numbers a lot.")
    st.write("Logins are not the same signal for everyone:")
    logins = history.groupby(np.where(history["international"] == 1, "International", "Domestic"))[
        ["logins_total", "logins_last3"]].mean().round(1)
    logins.columns = ["Logins, weeks 1-6", "Logins, last 3 weeks"]
    st.dataframe(logins, width="stretch")

# ---------- 4. the costs ----------
with tab4:
    st.write(f"With the costs in the sidebar, the best cut-off on the 2025 students is **{best_cut:.2f}**. "
             f"That would mean contacting **{int((val['risk'] >= best_cut).sum())}** students, "
             f"for a net value of **{max(values):,.0f} DKK**.")
    st.line_chart(pd.DataFrame({"cut-off": cuts, "net value (DKK)": values}).set_index("cut-off"))
    st.caption("Change the costs in the sidebar and watch the best cut-off move. These numbers are assumptions, "
               "and a person at the office should decide them.")
