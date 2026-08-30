from src.generation.generator import GroundedGenerator, GroundedAnswer

def test_pydantic_schema_structure():
    # Verify we can initialize and instantiate GroundedAnswer
    ans = GroundedAnswer(
        answer="The margin requirement for index futures is 12%.",
        citations=[
            {
                "source": "doc1.txt",
                "clause_id": "Clause 1.2",
                "clause_title": "Margin Composition",
                "snippet": "minimum initial margin requirement for Index Futures is 12%"
            }
        ],
        applicable_rules=["Initial Margin Requirements"],
        thresholds_and_timelines=["12% initial margin"],
        required_actions=[],
        grounding_confidence=1.0,
        is_sufficient=True
    )
    
    assert ans.is_sufficient is True
    assert len(ans.citations) == 1
    assert ans.citations[0].clause_id == "Clause 1.2"

def test_abstention_fallbacks():
    # If we pass empty context to the generator, it must return a refusal answer
    generator = GroundedGenerator()
    ans = generator.generate("What is the weather today?", [])
    
    assert ans.is_sufficient is False
    assert "cannot find information" in ans.answer or "Error" in ans.answer
    assert len(ans.citations) == 0
