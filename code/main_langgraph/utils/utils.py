import re

def longestCommonSubsequence(text1: str, text2: str) -> int:
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]

def add_reference(answer, searched_references):
    segments = re.split(r'[,，]', answer)

    result = []
    for segment in segments:
        subseq_len = 0
        subseq_ratio = 0.0
        ref_id = -1
        for ref_idx, reference in enumerate(searched_references):
            cand_subseq_len = longestCommonSubsequence(segment, reference)
            cand_ratio = float(cand_subseq_len)/len(segment)
            if cand_subseq_len > subseq_len:
                subseq_len = cand_subseq_len
                subseq_ratio = float(cand_subseq_len)/len(segment)
                ref_id = ref_idx+1

        result.append((segment, ref_id, subseq_ratio, subseq_len))

    return result

def render_answer_with_ref(answer_with_ref, ratio_threshold=0.65, len_threshold=6):
    filtered_segments = [ f"{seg[0]}<sup>[{seg[1]}]</sup>" for seg in answer_with_ref if seg[2] > ratio_threshold and seg[3] > len_threshold ]
    return ','.join(filtered_segments)


if __name__ == "__main__":
    answer = '根据提供的信息,AWS Clean Rooms 目前只支持 S3 数据源的接入,暂时还不支持其他数据源。'
    searched_references = [
        'Question: AWS Clean Rooms目前可以支持什么数据源的接入？ Answer: 目前只支持S3，其他数据源近期没有具体计划。',
        'Question: AWS Clean Rooms的数据源必须在AWS上么？ Answer: 对，目前必须在AWS上，而且必须是同一个region。',
        'Question: AWS Clean Rooms 与 AWS Data Exchange 是什么关系？ Answer: AWS Clean Rooms 可以通过AWS Data Exchange 去浏览和寻找可用数据的合作方。 他是AWS Data Exchange的更近一步的服务，提供了可控(多种约束限制)和可审计的数据合作方式。',
        'Question: AWS Clean Rooms能支持多大规模数据的查询？查询速度怎么样？ Answer: 能支持TB/GB级数据的查询。 一般查询延迟为几十秒到几分钟。默认计算容量为32 CRPUs, 目前这个默认计算容量不可设置，但是roadmap中未来打算让用户可以进行设置。(Slack中Ryan 提到，如果引擎中任务有积压，它能够scale up）'
    ]

    answer_with_ref = add_reference(answer, searched_references)
    print(render_answer_with_ref(answer_with_ref))

