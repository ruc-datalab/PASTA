# The following script is adapted from the script of Tapex.
# Original: https://github.com/microsoft/Table-Pretraining/blob/0b87efa253232d4aafa52c1f4725cb4f6e027877/tapex/processor/table_linearize.py

import abc
from typing import Dict, List

class TableLinearize(abc.ABC):

    PROMPT_MESSAGE = """
        Please check that your table must follow the following format:
        {"header": ["col1", "col2", "col3"], "rows": [["row11", "row12", "row13"], ["row21", "row22", "row23"]]}
    """

    def process_table(self, table_content: Dict) -> str:
        """
        Given a table, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        pass

    def process_header(self, headers: List):
        """
        Given a list of headers, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        pass

    def process_row(self, row: List, row_index: int):
        """
        Given a row, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        pass


class IndexedRowTableLinearize(TableLinearize):
    """
    FORMAT: [Header]: col1 | col2 | col3 [Row] : val1 | val2 | val3 [Row] : ...
    """

    def process_table(self, table_content: Dict):
        """
        Given a table, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        assert "header" in table_content and "rows" in table_content, self.PROMPT_MESSAGE
        # process header
        _table_str = self.process_header(table_content["header"]) + " "
        # process rows
        for i, row_example in enumerate(table_content["rows"]):
            # NOTE: the row should start from row 1 instead of 0
            _table_str += self.process_row(row_example, row_index=i + 1) + " "
        return _table_str.strip()

    def process_header(self, headers: List):
        """
        Given a list of headers, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        return "[Header] " + " | ".join(headers)

    def process_row(self, row: List, row_index: int):
        """
        Given a row, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        row_str = ""
        row_cell_values = []
        for cell_value in row:
            if isinstance(cell_value, int):
                row_cell_values.append(str(cell_value))
            else:
                row_cell_values.append(cell_value)
        row_str += " | ".join(row_cell_values)
        return "[Row] " + row_str