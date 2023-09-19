import pytest
from alpaca_farm.auto_annotations import PairwiseAutoAnnotator
import json


@pytest.mark.slow
def test_example():
    with open("examples/data/outputs_pairs.json") as f:
        outputs_pairs = json.load(f)[:6]
    annotator = PairwiseAutoAnnotator(is_avoid_reannotations=False, caching_path=None)
    annotated = annotator.annotate_pairs(outputs_pairs)
    expected_annotations = {'instruction': 'If you could help me write an email to my friends inviting them to '
                                            'dinner on Friday, it would be greatly appreciated.',
    'input': '',
    'output_1': "Dear Friends, \r\n\r\nI hope this message finds you well. I'm excited to invite you to dinner on "
                "Friday. We'll meet at 7:00 PM at [location]. I look forward to seeing you there. \r\n\r\nBest,\r\n[Name]",
    'output_2': "Hey everyone! \n\nI'm hosting a dinner party this Friday night and I'd love for all of you to come "
                "over. We'll have a delicious spread of food and some great conversations. \n\nLet me know if you can make it - I'd love to see you all there!\n\nCheers,\n[Your Name]",
    'annotator': 'chatgpt_2',
    'preference': 2}
    for k,v in expected_annotations.items():
        assert annotated[-1][k] == v
