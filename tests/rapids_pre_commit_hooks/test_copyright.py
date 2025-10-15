# Copyright (c) 2024, NVIDIA CORPORATION.
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

import contextlib
import datetime
import os.path
import tempfile
from textwrap import dedent
from unittest.mock import Mock, patch

import git
import pytest
from freezegun import freeze_time

from rapids_pre_commit_hooks import copyright
from rapids_pre_commit_hooks.lint import (
    Lines,
    Linter,
    LintWarning,
    Note,
    Replacement,
)


@pytest.mark.parametrize(
    ["content", "start", "expected_match"],
    [
        pytest.param(
            dedent(
                """
                Copyright (c) 2021 NVIDIA CORPORATION
                Copyright (c) 2025 NVIDIA CORPORATION
                """
            ),
            0,
            copyright.CopyrightMatch(
                span=(1, 38),
                spdx_filecopyrighttext_tag_span=None,
                full_copyright_text_span=(1, 38),
                nvidia_copyright_text_span=(1, 38),
                years_span=(15, 19),
                first_year_span=(15, 19),
                last_year_span=None,
                spdx_license_identifier_tag_span=None,
                spdx_license_identifier_text_span=None,
            ),
            id="basic-copyright-single-year",
        ),
        pytest.param(
            dedent(
                """
                # Copyright  (c)  2021-2025,  NVIDIA Corporation and affiliates
                """
            ),
            0,
            copyright.CopyrightMatch(
                span=(3, 64),
                spdx_filecopyrighttext_tag_span=None,
                full_copyright_text_span=(3, 64),
                nvidia_copyright_text_span=(3, 49),
                years_span=(19, 28),
                first_year_span=(19, 23),
                last_year_span=(24, 28),
                spdx_license_identifier_tag_span=None,
                spdx_license_identifier_text_span=None,
            ),
            id="basic-copyright-multi-year-with-extras",
        ),
        pytest.param(
            dedent(
                """
                Copyright (c) 2021 NVIDIA CORPORATION
                """
            ),
            2,
            None,
            id="basic-copyright-late-start-no-match",
        ),
        pytest.param(
            dedent(
                """
                Copyright (c) 2021 NVIDIA CORPORATION
                Copyright (c) 2025 NVIDIA CORPORATION
                """
            ),
            38,
            copyright.CopyrightMatch(
                span=(39, 76),
                spdx_filecopyrighttext_tag_span=None,
                full_copyright_text_span=(39, 76),
                nvidia_copyright_text_span=(39, 76),
                years_span=(53, 57),
                first_year_span=(53, 57),
                last_year_span=None,
                spdx_license_identifier_tag_span=None,
                spdx_license_identifier_text_span=None,
            ),
            id="basic-copyright-late-start-second-match",
        ),
        pytest.param(
            dedent(
                """
                # Copyright (c) 2021-2025, NVIDIA CORPORATION.
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
                """  # noqa: E501
            ),
            0,
            copyright.CopyrightMatch(
                span=(3, 593),
                spdx_filecopyrighttext_tag_span=None,
                full_copyright_text_span=(3, 47),
                nvidia_copyright_text_span=(3, 46),
                years_span=(17, 26),
                first_year_span=(17, 21),
                last_year_span=(22, 26),
                long_form_text_span=(49, 593),
                spdx_license_identifier_tag_span=None,
                spdx_license_identifier_text_span=None,
            ),
            id="basic-copyright-long-form-text",
        ),
        pytest.param(
            dedent(
                """
                # Copyright (c) 2021-2025, NVIDIA CORPORATION.
                #
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
                """  # noqa: E501
            ),
            0,
            copyright.CopyrightMatch(
                span=(3, 47),
                spdx_filecopyrighttext_tag_span=None,
                full_copyright_text_span=(3, 47),
                nvidia_copyright_text_span=(3, 46),
                years_span=(17, 26),
                first_year_span=(17, 21),
                last_year_span=(22, 26),
                long_form_text_span=None,
                spdx_license_identifier_tag_span=None,
                spdx_license_identifier_text_span=None,
            ),
            id="basic-copyright-wrong-long-form-text",
        ),
        pytest.param(
            dedent(
                """
                SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION
                """
            ),
            0,
            copyright.CopyrightMatch(
                span=(1, 62),
                spdx_filecopyrighttext_tag_span=(1, 25),
                full_copyright_text_span=(25, 62),
                nvidia_copyright_text_span=(25, 62),
                years_span=(39, 43),
                first_year_span=(39, 43),
                last_year_span=None,
                spdx_license_identifier_tag_span=None,
                spdx_license_identifier_text_span=None,
            ),
            id="spdx-copyright",
        ),
        pytest.param(
            dedent(
                """
                SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION
                SPDX-License-Identifier: Apache-2.0
                """
            ),
            0,
            copyright.CopyrightMatch(
                span=(1, 98),
                spdx_filecopyrighttext_tag_span=(1, 25),
                full_copyright_text_span=(25, 62),
                nvidia_copyright_text_span=(25, 62),
                years_span=(39, 43),
                first_year_span=(39, 43),
                last_year_span=None,
                spdx_license_identifier_tag_span=(63, 88),
                spdx_license_identifier_text_span=(88, 98),
            ),
            id="spdx-copyright-with-license-identifier",
        ),
        pytest.param(
            dedent(
                """
                SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION
                Licensed under the Apache License, Version 2.0 (the "License");
                you may not use this file except in compliance with the License.
                You may obtain a copy of the License at

                    http://www.apache.org/licenses/LICENSE-2.0

                Unless required by applicable law or agreed to in writing, software
                distributed under the License is distributed on an "AS IS" BASIS,
                WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
                See the License for the specific language governing permissions and
                limitations under the License.
                """  # noqa: E501
            ),
            0,
            copyright.CopyrightMatch(
                span=(1, 586),
                spdx_filecopyrighttext_tag_span=(1, 25),
                full_copyright_text_span=(25, 62),
                nvidia_copyright_text_span=(25, 62),
                years_span=(39, 43),
                first_year_span=(39, 43),
                last_year_span=None,
                long_form_text_span=(63, 586),
                spdx_license_identifier_tag_span=None,
                spdx_license_identifier_text_span=None,
            ),
            id="spdx-copyright-with-long-form-text",
        ),
        pytest.param(
            dedent(
                """
                SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION
                SPDX-License-Identifier: Apache-2.0

                Licensed under the Apache License, Version 2.0 (the "License");
                you may not use this file except in compliance with the License.
                You may obtain a copy of the License at

                    http://www.apache.org/licenses/LICENSE-2.0

                Unless required by applicable law or agreed to in writing, software
                distributed under the License is distributed on an "AS IS" BASIS,
                WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
                See the License for the specific language governing permissions and
                limitations under the License.
                """  # noqa: E501
            ),
            0,
            copyright.CopyrightMatch(
                span=(1, 623),
                spdx_filecopyrighttext_tag_span=(1, 25),
                full_copyright_text_span=(25, 62),
                nvidia_copyright_text_span=(25, 62),
                years_span=(39, 43),
                first_year_span=(39, 43),
                last_year_span=None,
                spdx_license_identifier_tag_span=(63, 88),
                spdx_license_identifier_text_span=(88, 98),
                long_form_text_span=(99, 623),
            ),
            id="spdx-copyright-with-license-identifier-and-long-form-text",
        ),
        pytest.param(
            dedent(
                """
                Copyright (c) 2021 NVIDIA CORPORATION
                SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION
                """
            ),
            0,
            copyright.CopyrightMatch(
                span=(1, 38),
                spdx_filecopyrighttext_tag_span=None,
                full_copyright_text_span=(1, 38),
                nvidia_copyright_text_span=(1, 38),
                years_span=(15, 19),
                first_year_span=(15, 19),
                last_year_span=None,
                spdx_license_identifier_tag_span=None,
                spdx_license_identifier_text_span=None,
            ),
            id="basic-copyright-and-spdx-copyright",
        ),
        pytest.param(
            dedent(
                """
                SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION
                Copyright (c) 2021 NVIDIA CORPORATION
                """
            ),
            0,
            copyright.CopyrightMatch(
                span=(1, 62),
                spdx_filecopyrighttext_tag_span=(1, 25),
                full_copyright_text_span=(25, 62),
                nvidia_copyright_text_span=(25, 62),
                years_span=(39, 43),
                first_year_span=(39, 43),
                last_year_span=None,
                spdx_license_identifier_tag_span=None,
                spdx_license_identifier_text_span=None,
            ),
            id="spdx-copyright-and-basic-copyright",
        ),
    ],
)
def test_match_copyright(content, start, expected_match):
    assert copyright.match_copyright(Lines(content), start) == expected_match


@pytest.mark.parametrize(
    ["content", "expected_matches"],
    [
        pytest.param(
            dedent(
                """
                Copyright (c) 2021 NVIDIA CORPORATION
                """
            ),
            [
                copyright.CopyrightMatch(
                    span=(1, 38),
                    spdx_filecopyrighttext_tag_span=None,
                    full_copyright_text_span=(1, 38),
                    nvidia_copyright_text_span=(1, 38),
                    years_span=(15, 19),
                    first_year_span=(15, 19),
                    last_year_span=None,
                    spdx_license_identifier_tag_span=None,
                    spdx_license_identifier_text_span=None,
                ),
            ],
            id="basic-copyright-single",
        ),
        pytest.param(
            dedent(
                """
                Copyright (c) 2021 NVIDIA CORPORATION
                Copyright (c) 2025 NVIDIA CORPORATION
                """
            ),
            [
                copyright.CopyrightMatch(
                    span=(1, 38),
                    spdx_filecopyrighttext_tag_span=None,
                    full_copyright_text_span=(1, 38),
                    nvidia_copyright_text_span=(1, 38),
                    years_span=(15, 19),
                    first_year_span=(15, 19),
                    last_year_span=None,
                    spdx_license_identifier_tag_span=None,
                    spdx_license_identifier_text_span=None,
                ),
                copyright.CopyrightMatch(
                    span=(39, 76),
                    spdx_filecopyrighttext_tag_span=None,
                    full_copyright_text_span=(39, 76),
                    nvidia_copyright_text_span=(39, 76),
                    years_span=(53, 57),
                    first_year_span=(53, 57),
                    last_year_span=None,
                    spdx_license_identifier_tag_span=None,
                    spdx_license_identifier_text_span=None,
                ),
            ],
            id="basic-copyright-multiple",
        ),
        pytest.param(
            dedent(
                """
                SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION
                SPDX-License-Identifier: Apache-2.0

                Licensed under the Apache License, Version 2.0 (the "License");
                you may not use this file except in compliance with the License.
                You may obtain a copy of the License at

                    http://www.apache.org/licenses/LICENSE-2.0

                Unless required by applicable law or agreed to in writing, software
                distributed under the License is distributed on an "AS IS" BASIS,
                WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
                See the License for the specific language governing permissions and
                limitations under the License.

                SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION
                SPDX-License-Identifier: Apache-2.0

                Licensed under the Apache License, Version 2.0 (the "License");
                you may not use this file except in compliance with the License.
                You may obtain a copy of the License at

                    http://www.apache.org/licenses/LICENSE-2.0

                Unless required by applicable law or agreed to in writing, software
                distributed under the License is distributed on an "AS IS" BASIS,
                WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
                See the License for the specific language governing permissions and
                limitations under the License.
                """  # noqa: E501
            ),
            [
                copyright.CopyrightMatch(
                    span=(1, 623),
                    spdx_filecopyrighttext_tag_span=(1, 25),
                    full_copyright_text_span=(25, 62),
                    nvidia_copyright_text_span=(25, 62),
                    years_span=(39, 43),
                    first_year_span=(39, 43),
                    last_year_span=None,
                    spdx_license_identifier_tag_span=(63, 88),
                    spdx_license_identifier_text_span=(88, 98),
                    long_form_text_span=(99, 623),
                ),
                copyright.CopyrightMatch(
                    span=(625, 1247),
                    spdx_filecopyrighttext_tag_span=(625, 649),
                    full_copyright_text_span=(649, 686),
                    nvidia_copyright_text_span=(649, 686),
                    years_span=(663, 667),
                    first_year_span=(663, 667),
                    last_year_span=None,
                    spdx_license_identifier_tag_span=(687, 712),
                    spdx_license_identifier_text_span=(712, 722),
                    long_form_text_span=(723, 1247),
                ),
            ],
            id="spdx-copyright-multiple-with-long-form-text",
        ),
        pytest.param(
            "Hello world",
            [],
            id="no-copyright",
        ),
    ],
)
def test_match_all_copyright(content, expected_matches):
    assert (
        list(copyright.match_all_copyright(Lines(content))) == expected_matches
    )


@pytest.mark.parametrize(
    ["content", "index", "expected_pos"],
    [
        pytest.param(
            dedent(
                r"""
                # SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION
                # SPDX-License-Identifier: Apache-2.0
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
                This is a file
                """  # noqa: E501
            ),
            67,
            (104, 648),
            id="correct",
        ),
        pytest.param(
            dedent(
                r"""
                # SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION
                # SPDX-License-Identifier: Apache-2.0
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
                This is a file
                """  # noqa: E501
            ),
            3,
            None,
            id="wrong-start-line",
        ),
        pytest.param(
            dedent(
                r"""
                # SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION
                # SPDX-License-Identifier: Apache-2.0
                #
                # Licensed under the Apache License, Version 2.0 (the "License");
                # you may not use this file except in compliance with the License.
                # You may obtain a copy of the License at
                *
                #     http://www.apache.org/licenses/LICENSE-2.0
                #
                # Unless required by applicable law or agreed to in writing, software
                # distributed under the License is distributed on an "AS IS" BASIS,
                # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
                # See the License for the specific language governing permissions and
                # limitations under the License.
                This is a file
                """  # noqa: E501
            ),
            67,
            None,
            id="mismatched-prefix",
        ),
        pytest.param(
            dedent(
                r"""
                # SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION
                # SPDX-License-Identifier: Apache-2.0
                #
                # Licensed under the Apache License, Version 2.0 (the "License");
                # you may not use this file except in compliance with the License.
                # You may obtain a copy of the License at:
                #
                #     http://www.apache.org/licenses/LICENSE-2.0
                #
                # Unless required by applicable law or agreed to in writing, software
                # distributed under the License is distributed on an "AS IS" BASIS,
                # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
                # See the License for the specific language governing permissions and
                # limitations under the License.
                This is a file
                """  # noqa: E501
            ),
            67,
            None,
            id="mismatched-contents",
        ),
        pytest.param(
            dedent(
                r"""
                # SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION
                # SPDX-License-Identifier: Apache-2.0
                #
                # Licensed under the Apache License, Version 2.0 (the "License");
                # you may not use this file except in compliance with the License.
                # You may obtain a copy of the License at:
                #
                #     http://www.apache.org/licenses/LICENSE-2.0
                """  # noqa: E501
            ),
            67,
            None,
            id="too-short",
        ),
    ],
)
def test_find_long_form_text(content, index, expected_pos):
    assert (
        copyright.find_long_form_text(Lines(content), "Apache-2.0", index)
        == expected_pos
    )


@pytest.mark.parametrize(
    ["content", "expected_stripped"],
    [
        pytest.param(
            dedent(
                """
                This is a line before the first copyright statement
                Copyright (c) 2024 NVIDIA CORPORATION
                This is a line between the first two copyright statements
                Copyright (c) 2021-2024 NVIDIA CORPORATION
                This is a line between the next two copyright statements
                # Copyright 2021,  NVIDIA Corporation and affiliates
                This is a line after the last copyright statement
                """
            ),
            [
                "\nThis is a line before the first copyright statement\n",
                "\nThis is a line between the first two copyright statements"
                "\n",
                "\nThis is a line between the next two copyright statements"
                "\n# ",
                "\nThis is a line after the last copyright statement\n",
            ],
            id="basic-copyright",
        ),
        pytest.param(
            "No copyright here",
            ["No copyright here"],
            id="no-copyright",
        ),
        pytest.param(
            dedent(
                """
                # SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION
                # SPDX-License-Identifier: Apache-2.0
                This is a line after the license identifier
                """
            ),
            [
                "\n# ",
                "\nThis is a line after the license identifier\n",
            ],
            id="spdx",
        ),
        pytest.param(
            dedent(
                """
                # Copyright (c) 2024 NVIDIA CORPORATION
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
                This is a line after the license text
                """  # noqa: E501
            ),
            [
                "\n# ",
                "\nThis is a line after the license text\n",
            ],
            id="basic-long-form-text",
        ),
        pytest.param(
            dedent(
                """
                # SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION
                # SPDX-License-Identifier: Apache-2.0
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
                This is a line after the license text
                """  # noqa: E501
            ),
            [
                "\n# ",
                "\nThis is a line after the license text\n",
            ],
            id="spdx-long-form-text",
        ),
        pytest.param(
            dedent(
                """
                # SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION
                # SPDX-License-Identifier: Apache-2.0
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
                This is a line after the license text
                # SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION
                # SPDX-License-Identifier: Apache-2.0
                """  # noqa: E501
            ),
            [
                "\n# ",
                "\nThis is a line after the license text\n# ",
                "\n",
            ],
            id="spdx-long-form-text-multiple-headers",
        ),
    ],
)
def test_strip_copyright(content, expected_stripped):
    lines = Lines(content)
    matches = copyright.match_all_copyright(lines)
    assert (
        copyright.strip_copyright(
            lines,
            matches,
        )
        == expected_stripped
    )


@pytest.mark.parametrize(
    [
        "change_type",
        "old_filename",
        "old_content",
        "new_filename",
        "new_content",
        "spdx",
        "force_spdx",
        "warnings",
    ],
    [
        pytest.param(
            "A",
            None,
            None,
            "file.txt",
            "No copyright notice",
            False,
            False,
            [
                LintWarning((0, 0), "no copyright notice found"),
            ],
            id="added-with-no-copyright-notice",
        ),
        pytest.param(
            "M",
            "file.txt",
            "No copyright notice",
            "file.txt",
            "No copyright notice",
            False,
            False,
            [],
            id="unchanged-with-no-copyright-notice",
        ),
        pytest.param(
            "M",
            "file.txt",
            dedent(
                """
                Copyright (c) 2021-2023 NVIDIA CORPORATION
                Copyright (c) 2023 NVIDIA CORPORATION
                Copyright (c) 2024 NVIDIA CORPORATION
                Copyright (c) 2025 NVIDIA CORPORATION
                This file has not been changed
                """
            ),
            "file.txt",
            dedent(
                """
                Copyright (c) 2021-2023 NVIDIA CORPORATION
                Copyright (c) 2023 NVIDIA CORPORATION
                Copyright (c) 2024 NVIDIA CORPORATION
                Copyright (c) 2025 NVIDIA CORPORATION
                This file has not been changed
                """
            ),
            False,
            False,
            [],
            id="unchanged-with-copyright-notice",
        ),
        pytest.param(
            "M",
            "file.txt",
            dedent(
                """
                Copyright (c) 2021-2023 NVIDIA CORPORATION
                Copyright (c) 2023 NVIDIA CORPORATION
                This file has not been changed
                """
            ),
            "file.txt",
            dedent(
                """
                Copyright (c) 2021-2023 NVIDIA CORPORATION
                Copyright (c) 2023 NVIDIA CORPORATION
                This file has been changed
                """
            ),
            False,
            False,
            [
                LintWarning(
                    (58, 62),
                    "copyright is out of date",
                    replacements=[
                        Replacement(
                            (44, 81),
                            "Copyright (c) 2023-2024, NVIDIA CORPORATION",
                        ),
                    ],
                ),
            ],
            id="changed-with-no-copyright-update",
        ),
        pytest.param(
            "M",
            "file.txt",
            dedent(
                """
                Copyright (c) 2021-2023 NVIDIA CORPORATION
                Copyright (c) 2023 NVIDIA CORPORATION
                Copyright (c) 2024 NVIDIA CORPORATION
                This file has not been changed
                """
            ),
            "file.txt",
            dedent(
                """
                Copyright (c) 2021-2023 NVIDIA CORPORATION
                Copyright (c) 2023 NVIDIA CORPORATION
                Copyright (c) 2024 NVIDIA CORPORATION
                This file has been changed
                """
            ),
            False,
            False,
            [],
            id="changed-with-no-copyright-update-newest-up-to-date",
        ),
        pytest.param(
            "A",
            None,
            None,
            "file.txt",
            dedent(
                """
                Copyright (c) 2021-2023 NVIDIA CORPORATION
                Copyright (c) 2023 NVIDIA CORPORATION
                This file has been changed
                """
            ),
            False,
            False,
            [
                LintWarning(
                    (58, 62),
                    "copyright is out of date",
                    replacements=[
                        Replacement(
                            (44, 81),
                            "Copyright (c) 2023-2024, NVIDIA CORPORATION",
                        ),
                    ],
                ),
            ],
            id="added-with-out-of-date-copyright",
        ),
        pytest.param(
            "M",
            "file.txt",
            dedent(
                """
                Copyright (c) 2021-2023 NVIDIA CORPORATION
                Copyright (c) 2023 NVIDIA CORPORATION
                Copyright (c) 2024 NVIDIA CORPORATION
                Copyright (c) 2025 NVIDIA CORPORATION
                This file has not been changed
                """
            ),
            "file.txt",
            dedent(
                """
                Copyright (c) 2021-2024 NVIDIA CORPORATION
                Copyright (c) 2023 NVIDIA CORPORATION
                Copyright (c) 2024 NVIDIA CORPORATION
                Copyright (c) 2025 NVIDIA Corporation
                This file has not been changed
                """
            ),
            False,
            False,
            [
                LintWarning(
                    (15, 24),
                    "copyright is not out of date and should not be updated",
                    replacements=[
                        Replacement(
                            (1, 43),
                            "Copyright (c) 2021-2023 NVIDIA CORPORATION",
                        ),
                    ],
                ),
                LintWarning(
                    (120, 157),
                    "copyright is not out of date and should not be updated",
                    replacements=[
                        Replacement(
                            (120, 157), "Copyright (c) 2025 NVIDIA CORPORATION"
                        ),
                    ],
                ),
            ],
            id="unchanged-with-copyright-update",
        ),
        pytest.param(
            "R",
            "file1.txt",
            dedent(
                """
                Copyright (c) 2021-2023 NVIDIA CORPORATION
                Copyright (c) 2023 NVIDIA CORPORATION
                This file has not been changed
                """
            ),
            "file2.txt",
            dedent(
                """
                Copyright (c) 2021-2023 NVIDIA CORPORATION
                Copyright (c) 2023 NVIDIA CORPORATION
                This file has been changed
                """
            ),
            False,
            False,
            [
                LintWarning(
                    (58, 62),
                    "copyright is out of date",
                    replacements=[
                        Replacement(
                            (44, 81),
                            "Copyright (c) 2023-2024, NVIDIA CORPORATION",
                        ),
                    ],
                ),
            ],
            id="renamed-and-changed-with-no-copyright-update",
        ),
        pytest.param(
            "C",
            "file1.txt",
            dedent(
                """
                Copyright (c) 2021-2023 NVIDIA CORPORATION
                Copyright (c) 2023 NVIDIA CORPORATION
                This file has not been changed
                """
            ),
            "file2.txt",
            dedent(
                """
                Copyright (c) 2021-2023 NVIDIA CORPORATION
                Copyright (c) 2023 NVIDIA CORPORATION
                This file has been changed
                """
            ),
            False,
            False,
            [
                LintWarning(
                    (58, 62),
                    "copyright is out of date",
                    replacements=[
                        Replacement(
                            (44, 81),
                            "Copyright (c) 2023-2024, NVIDIA CORPORATION",
                        ),
                    ],
                ),
            ],
            id="copied-and-changed-with-no-copyright-update",
        ),
        pytest.param(
            "R",
            "file1.txt",
            dedent(
                """
                Copyright (c) 2021-2023 NVIDIA CORPORATION
                Copyright (c) 2023 NVIDIA CORPORATION
                Copyright (c) 2024 NVIDIA CORPORATION
                Copyright (c) 2025 NVIDIA CORPORATION
                This file has not been changed
                """
            ),
            "file2.txt",
            dedent(
                """
                Copyright (c) 2024 NVIDIA CORPORATION
                Copyright (c) 2023-2024 NVIDIA CORPORATION
                Copyright (c) 2024 NVIDIA CORPORATION
                Copyright (c) 2025 NVIDIA CORPORATION
                This file has been changed
                """
            ),
            False,
            False,
            [],
            id="renamed-and-changed-with-copyright-update",
        ),
        pytest.param(
            "C",
            "file1.txt",
            dedent(
                """
                Copyright (c) 2021-2023 NVIDIA CORPORATION
                Copyright (c) 2023 NVIDIA CORPORATION
                Copyright (c) 2024 NVIDIA CORPORATION
                Copyright (c) 2025 NVIDIA CORPORATION
                This file has not been changed
                """
            ),
            "file2.txt",
            dedent(
                """
                Copyright (c) 2024 NVIDIA CORPORATION
                Copyright (c) 2023-2024 NVIDIA CORPORATION
                Copyright (c) 2024 NVIDIA CORPORATION
                Copyright (c) 2025 NVIDIA CORPORATION
                This file has been changed
                """
            ),
            False,
            False,
            [],
            id="copied-and-changed-with-copyright-update",
        ),
        pytest.param(
            "R",
            "file1.txt",
            dedent(
                """
                Copyright (c) 2021-2023 NVIDIA CORPORATION
                Copyright (c) 2023 NVIDIA CORPORATION
                Copyright (c) 2024 NVIDIA CORPORATION
                Copyright (c) 2025 NVIDIA CORPORATION
                This file has not been changed
                """
            ),
            "file2.txt",
            dedent(
                """
                Copyright (c) 2021-2024 NVIDIA CORPORATION
                Copyright (c) 2023 NVIDIA CORPORATION
                Copyright (c) 2024 NVIDIA CORPORATION
                Copyright (c) 2025 NVIDIA Corporation
                This file has not been changed
                """
            ),
            False,
            False,
            [
                LintWarning(
                    (15, 24),
                    "copyright is not out of date and should not be updated",
                    notes=[
                        Note(
                            (0, 189),
                            "file was renamed from 'file1.txt' and is assumed "
                            "to share history with it",
                        ),
                        Note(
                            (0, 189),
                            "change file contents if you want its copyright "
                            "dates to only be determined by its own edit "
                            "history",
                        ),
                    ],
                    replacements=[
                        Replacement(
                            (1, 43),
                            "Copyright (c) 2021-2023 NVIDIA CORPORATION",
                        ),
                    ],
                ),
                LintWarning(
                    (120, 157),
                    "copyright is not out of date and should not be updated",
                    notes=[
                        Note(
                            (0, 189),
                            "file was renamed from 'file1.txt' and is assumed "
                            "to share history with it",
                        ),
                        Note(
                            (0, 189),
                            "change file contents if you want its copyright "
                            "dates to only be determined by its own edit "
                            "history",
                        ),
                    ],
                    replacements=[
                        Replacement(
                            (120, 157), "Copyright (c) 2025 NVIDIA CORPORATION"
                        ),
                    ],
                ),
            ],
            id="renamed-and-unchanged-with-copyright-update",
        ),
        pytest.param(
            "C",
            "file1.txt",
            dedent(
                """
                Copyright (c) 2021-2023 NVIDIA CORPORATION
                Copyright (c) 2023 NVIDIA CORPORATION
                Copyright (c) 2024 NVIDIA CORPORATION
                Copyright (c) 2025 NVIDIA CORPORATION
                This file has not been changed
                """
            ),
            "file2.txt",
            dedent(
                """
                Copyright (c) 2021-2024 NVIDIA CORPORATION
                Copyright (c) 2023 NVIDIA CORPORATION
                Copyright (c) 2024 NVIDIA CORPORATION
                Copyright (c) 2025 NVIDIA Corporation
                This file has not been changed
                """
            ),
            False,
            False,
            [
                LintWarning(
                    (15, 24),
                    "copyright is not out of date and should not be updated",
                    notes=[
                        Note(
                            (0, 189),
                            "file was copied from 'file1.txt' and is assumed "
                            "to share history with it",
                        ),
                        Note(
                            (0, 189),
                            "change file contents if you want its copyright "
                            "dates to only be determined by its own edit "
                            "history",
                        ),
                    ],
                    replacements=[
                        Replacement(
                            (1, 43),
                            "Copyright (c) 2021-2023 NVIDIA CORPORATION",
                        ),
                    ],
                ),
                LintWarning(
                    (120, 157),
                    "copyright is not out of date and should not be updated",
                    notes=[
                        Note(
                            (0, 189),
                            "file was copied from 'file1.txt' and is assumed "
                            "to share history with it",
                        ),
                        Note(
                            (0, 189),
                            "change file contents if you want its copyright "
                            "dates to only be determined by its own edit "
                            "history",
                        ),
                    ],
                    replacements=[
                        Replacement(
                            (120, 157), "Copyright (c) 2025 NVIDIA CORPORATION"
                        ),
                    ],
                ),
            ],
            id="copied-and-unchanged-with-copyright-update",
        ),
        pytest.param(
            "M",
            "file1.txt",
            dedent(
                """
                SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION
                SPDX-License-Identifier: Apache-2.0
                This file has not been changed
                """
            ),
            "file1.txt",
            dedent(
                """
                SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION
                SPDX-License-Identifier: Apache-2.0
                This file has been changed
                """
            ),
            True,
            False,
            [],
            id="spdx-changed-with-headers",
        ),
        pytest.param(
            "M",
            "file1.txt",
            dedent(
                """
                Copyright (c) 2024 NVIDIA CORPORATION
                Copyright (c) 2023 NVIDIA CORPORATION
                This file has not been changed
                """
            ),
            "file1.txt",
            dedent(
                """
                Copyright (c) 2024 NVIDIA CORPORATION
                Copyright (c) 2023 NVIDIA CORPORATION
                This file has been changed
                """
            ),
            True,
            False,
            [
                LintWarning(
                    (1, 38),
                    "include SPDX-FileCopyrightText header",
                    replacements=[
                        Replacement(
                            (1, 1),
                            "SPDX-FileCopyrightText: ",
                        ),
                    ],
                ),
                LintWarning(
                    (1, 38),
                    "no SPDX-License-Identifier header found",
                    replacements=[
                        Replacement(
                            (38, 38),
                            "\nSPDX-License-Identifier: Apache-2.0",
                        ),
                    ],
                ),
            ],
            id="spdx-changed-with-no-headers",
        ),
        pytest.param(
            "M",
            "file1.txt",
            dedent(
                """
                Copyright (c) 2023 NVIDIA CORPORATION
                This file has not been changed
                """
            ),
            "file1.txt",
            dedent(
                """
                Copyright (c) 2023 NVIDIA CORPORATION
                This file has been changed
                """
            ),
            True,
            False,
            [
                LintWarning(
                    (15, 19),
                    "copyright is out of date",
                    replacements=[
                        Replacement(
                            (1, 38),
                            "Copyright (c) 2023-2024, NVIDIA CORPORATION",
                        ),
                    ],
                ),
                LintWarning(
                    (1, 38),
                    "include SPDX-FileCopyrightText header",
                    replacements=[
                        Replacement(
                            (1, 1),
                            "SPDX-FileCopyrightText: ",
                        ),
                    ],
                ),
                LintWarning(
                    (1, 38),
                    "no SPDX-License-Identifier header found",
                    replacements=[
                        Replacement(
                            (38, 38),
                            "\nSPDX-License-Identifier: Apache-2.0",
                        ),
                    ],
                ),
            ],
            id="spdx-changed-with-no-headers-and-out-of-date-copyright",
        ),
        pytest.param(
            "M",
            "file1.txt",
            dedent(
                """
                SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION
                SPDX-License-Identifier: BSD-3-Clause
                This file has not been changed
                """
            ),
            "file1.txt",
            dedent(
                """
                SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION
                SPDX-License-Identifier: BSD-3-Clause
                This file has been changed
                """
            ),
            True,
            False,
            [
                LintWarning(
                    (63, 100),
                    "SPDX-License-Identifier is incorrect",
                    replacements=[
                        Replacement(
                            (88, 100),
                            "Apache-2.0",
                        ),
                    ],
                ),
            ],
            id="spdx-changed-with-headers-and-incorrect-license-identifier",
        ),
        pytest.param(
            "M",
            "file1.txt",
            dedent(
                """
                Copyright (c) 2024 NVIDIA CORPORATION
                This file has not been changed
                """
            ),
            "file1.txt",
            dedent(
                """
                Copyright (c) 2024 NVIDIA CORPORATION
                This file has not been changed
                """
            ),
            True,
            False,
            [],
            id="spdx-unchanged-with-no-headers",
        ),
        pytest.param(
            "M",
            "file1.txt",
            dedent(
                """
                // Copyright (c) 2024 NVIDIA CORPORATION
                // Copyright (c) 2023 NVIDIA CORPORATION
                // This file has not been changed
                """
            ),
            "file1.txt",
            dedent(
                """
                // Copyright (c) 2024 NVIDIA CORPORATION
                // Copyright (c) 2023 NVIDIA CORPORATION
                // This file has not been changed
                """
            ),
            False,
            True,
            [
                LintWarning(
                    (4, 41),
                    "include SPDX-FileCopyrightText header",
                    replacements=[
                        Replacement(
                            (4, 4),
                            "SPDX-FileCopyrightText: ",
                        ),
                    ],
                ),
                LintWarning(
                    (4, 41),
                    "no SPDX-License-Identifier header found",
                    replacements=[
                        Replacement(
                            (41, 41),
                            "\n// SPDX-License-Identifier: Apache-2.0",
                        ),
                    ],
                ),
            ],
            id="force-spdx-unchanged-with-comments-and-no-headers",
        ),
        pytest.param(
            "M",
            "file1.cpp",
            dedent(
                """
                /* SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION
                 */
                This file has not been changed
                """  # noqa: E501
            ),
            "file1.cpp",
            dedent(
                """
                /* SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION
                 */
                This file has been changed
                """  # noqa: E501
            ),
            True,
            False,
            [
                LintWarning(
                    (4, 65),
                    "no SPDX-License-Identifier header found",
                    replacements=[
                        Replacement(
                            (65, 65),
                            "\n * SPDX-License-Identifier: Apache-2.0",
                        ),
                    ],
                ),
            ],
            id="spdx-changed-with-c-style-comments-and-no-license-header",
        ),
        pytest.param(
            "M",
            "file1.txt",
            dedent(
                """
                Copyright (c) 2023 NVIDIA CORPORATION
                This file has not been changed
                """
            ),
            "file1.txt",
            dedent(
                """
                SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION
                SPDX-License-Identifier: Apache-2.0
                This file has not been changed
                """
            ),
            True,
            False,
            [],
            id="spdx-headers-added-and-no-other-changes",
        ),
        pytest.param(
            "M",
            "file1.txt",
            dedent(
                """
                # SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION
                # SPDX-License-Identifier: Apache-2.0
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
                This file has not been changed
                """  # noqa: E501
            ),
            "file1.txt",
            dedent(
                """
                # SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION
                # SPDX-License-Identifier: Apache-2.0
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
                This file has not been changed
                """  # noqa: E501
            ),
            False,
            True,
            [
                LintWarning(
                    (104, 648),
                    "remove long-form copyright text",
                    replacements=[
                        Replacement((102, 648), ""),
                    ],
                ),
            ],
            id="force-spdx-with-headers-and-long-form-text",
        ),
        pytest.param(
            "M",
            "file1.txt",
            dedent(
                """
                # SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION
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
                This file has not been changed
                """  # noqa: E501
            ),
            "file1.txt",
            dedent(
                """
                # SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION
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
                This file has not been changed
                """  # noqa: E501
            ),
            False,
            True,
            [
                LintWarning(
                    (3, 64),
                    "no SPDX-License-Identifier header found",
                    replacements=[
                        Replacement(
                            (64, 64), "\n# SPDX-License-Identifier: Apache-2.0"
                        ),
                    ],
                ),
                LintWarning(
                    (66, 610),
                    "remove long-form copyright text",
                    replacements=[
                        Replacement((64, 610), ""),
                    ],
                ),
            ],
            id="force-spdx-unchanged-with-no-headers-and-long-form-text",
        ),
        pytest.param(
            "M",
            "file1.txt",
            dedent(
                """
                # SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION
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
                This file has not been changed
                """  # noqa: E501
            ),
            "file1.txt",
            dedent(
                """
                # SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION
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
                This file has been changed
                """  # noqa: E501
            ),
            False,
            True,
            [
                LintWarning(
                    (3, 70),
                    "no SPDX-License-Identifier header found",
                    replacements=[
                        Replacement(
                            (70, 70), "\n# SPDX-License-Identifier: Apache-2.0"
                        ),
                    ],
                ),
                LintWarning(
                    (72, 616),
                    "remove long-form copyright text",
                    replacements=[
                        Replacement((70, 616), ""),
                    ],
                ),
            ],
            id="force-spdx-changed-with-no-identifier-and-long-form-text",
        ),
    ],
)
@freeze_time("2024-01-18")
def test_apply_copyright_check(
    change_type,
    old_filename,
    old_content,
    new_filename,
    new_content,
    spdx,
    force_spdx,
    warnings,
):
    linter = Linter(new_filename, new_content)
    mock_args = Mock(
        spdx=spdx, force_spdx=force_spdx, spdx_license_identifier="Apache-2.0"
    )
    copyright.apply_copyright_check(
        linter, mock_args, change_type, old_filename, old_content
    )
    assert linter.warnings == warnings


@pytest.fixture
def git_repo(tmp_path):
    repo = git.Repo.init(tmp_path)
    with repo.config_writer() as w:
        w.set_value("user", "name", "RAPIDS Test Fixtures")
        w.set_value("user", "email", "testfixtures@rapids.ai")
    return repo


def test_get_target_branch(git_repo):
    with patch.dict("os.environ", {}, clear=True):
        args = Mock(main_branch=None, target_branch=None)

        with open(
            os.path.join(git_repo.working_tree_dir, "file.txt"), "w"
        ) as f:
            f.write("File\n")
        git_repo.index.add(["file.txt"])
        git_repo.index.commit("Initial commit")
        with pytest.warns(
            copyright.NoTargetBranchWarning,
            match=r"^Could not determine target branch[.] Try setting the "
            r"TARGET_BRANCH environment variable, or setting the "
            r"rapidsai[.]baseBranch configuration option[.]$",
        ):
            assert copyright.get_target_branch(git_repo, args) is None

        git_repo.create_head("branch-24.02")
        assert copyright.get_target_branch(git_repo, args) == "branch-24.02"

        args.main_branch = ""
        args.target_branch = ""

        git_repo.create_head("branch-24.04")
        git_repo.create_head("branch-24.03")
        assert copyright.get_target_branch(git_repo, args) == "branch-24.04"

        git_repo.create_head("branch-25.01")
        assert copyright.get_target_branch(git_repo, args) == "branch-25.01"

        args.main_branch = "main"
        assert copyright.get_target_branch(git_repo, args) == "main"

        with git_repo.config_writer() as w:
            w.set_value("rapidsai", "baseBranch", "nonexistent")
        assert copyright.get_target_branch(git_repo, args) == "nonexistent"

        with git_repo.config_writer() as w:
            w.set_value("rapidsai", "baseBranch", "branch-24.03")
        assert copyright.get_target_branch(git_repo, args) == "branch-24.03"

        with patch.dict("os.environ", {"RAPIDS_BASE_BRANCH": "nonexistent"}):
            assert copyright.get_target_branch(git_repo, args) == "nonexistent"

        with patch.dict("os.environ", {"RAPIDS_BASE_BRANCH": "master"}):
            assert copyright.get_target_branch(git_repo, args) == "master"

        with patch.dict(
            "os.environ",
            {"GITHUB_BASE_REF": "nonexistent", "RAPIDS_BASE_BRANCH": "master"},
        ):
            assert copyright.get_target_branch(git_repo, args) == "nonexistent"

        with patch.dict(
            "os.environ",
            {
                "GITHUB_BASE_REF": "branch-24.02",
                "RAPIDS_BASE_BRANCH": "master",
            },
        ):
            assert (
                copyright.get_target_branch(git_repo, args) == "branch-24.02"
            )

        with patch.dict(
            "os.environ",
            {
                "GITHUB_BASE_REF": "branch-24.02",
                "RAPIDS_BASE_BRANCH": "master",
                "TARGET_BRANCH": "nonexistent",
            },
        ):
            assert copyright.get_target_branch(git_repo, args) == "nonexistent"

        with patch.dict(
            "os.environ",
            {
                "GITHUB_BASE_REF": "branch-24.02",
                "RAPIDS_BASE_BRANCH": "master",
                "TARGET_BRANCH": "branch-24.04",
            },
        ):
            assert (
                copyright.get_target_branch(git_repo, args) == "branch-24.04"
            )
            args.target_branch = "nonexistent"
            assert copyright.get_target_branch(git_repo, args) == "nonexistent"
            args.target_branch = "master"
            assert copyright.get_target_branch(git_repo, args) == "master"


def test_get_target_branch_upstream_commit(git_repo):
    def fn(repo, filename):
        return os.path.join(repo.working_tree_dir, filename)

    def write_file(repo, filename, contents):
        with open(fn(repo, filename), "w") as f:
            f.write(contents)

    def mock_target_branch(branch):
        return patch(
            "rapids_pre_commit_hooks.copyright.get_target_branch",
            Mock(return_value=branch),
        )

    # fmt: off
    with tempfile.TemporaryDirectory() as remote_dir_1, \
         tempfile.TemporaryDirectory() as remote_dir_2:
        # fmt: on
        remote_repo_1 = git.Repo.init(remote_dir_1)
        remote_repo_2 = git.Repo.init(remote_dir_2)

        remote_1_master = remote_repo_1.head.reference

        write_file(remote_repo_1, "file1.txt", "File 1")
        write_file(remote_repo_1, "file2.txt", "File 2")
        write_file(remote_repo_1, "file3.txt", "File 3")
        write_file(remote_repo_1, "file4.txt", "File 4")
        write_file(remote_repo_1, "file5.txt", "File 5")
        write_file(remote_repo_1, "file6.txt", "File 6")
        write_file(remote_repo_1, "file7.txt", "File 7")
        remote_repo_1.index.add(
            [
                "file1.txt",
                "file2.txt",
                "file3.txt",
                "file4.txt",
                "file5.txt",
                "file6.txt",
                "file7.txt",
            ]
        )
        remote_repo_1.index.commit("Initial commit")

        remote_1_branch_1 = remote_repo_1.create_head(
            "branch-1-renamed", remote_1_master.commit
        )
        remote_repo_1.head.reference = remote_1_branch_1
        remote_repo_1.head.reset(index=True, working_tree=True)
        write_file(remote_repo_1, "file1.txt", "File 1 modified")
        remote_repo_1.index.add(["file1.txt"])
        remote_repo_1.index.commit(
            "Update file1.txt",
            commit_date=datetime.datetime(
                2024, 2, 1, tzinfo=datetime.timezone.utc,
            ),
        )

        remote_1_branch_2 = remote_repo_1.create_head(
            "branch-2", remote_1_master.commit
        )
        remote_repo_1.head.reference = remote_1_branch_2
        remote_repo_1.head.reset(index=True, working_tree=True)
        write_file(remote_repo_1, "file2.txt", "File 2 modified")
        remote_repo_1.index.add(["file2.txt"])
        remote_repo_1.index.commit("Update file2.txt")

        remote_1_branch_3 = remote_repo_1.create_head(
            "branch-3", remote_1_master.commit
        )
        remote_repo_1.head.reference = remote_1_branch_3
        remote_repo_1.head.reset(index=True, working_tree=True)
        write_file(remote_repo_1, "file3.txt", "File 3 modified")
        remote_repo_1.index.add(["file3.txt"])
        remote_repo_1.index.commit(
            "Update file3.txt",
            commit_date=datetime.datetime(
                2025, 1, 1, tzinfo=datetime.timezone.utc,
            ),
        )

        remote_1_branch_4 = remote_repo_1.create_head(
            "branch-4", remote_1_master.commit
        )
        remote_repo_1.head.reference = remote_1_branch_4
        remote_repo_1.head.reset(index=True, working_tree=True)
        write_file(remote_repo_1, "file4.txt", "File 4 modified")
        remote_repo_1.index.add(["file4.txt"])
        remote_repo_1.index.commit(
            "Update file4.txt",
            commit_date=datetime.datetime(
                2024, 1, 1, tzinfo=datetime.timezone.utc,
            ),
        )

        remote_1_branch_7 = remote_repo_1.create_head(
            "branch-7", remote_1_master.commit
        )
        remote_repo_1.head.reference = remote_1_branch_7
        remote_repo_1.head.reset(index=True, working_tree=True)
        write_file(remote_repo_1, "file7.txt", "File 7 modified")
        remote_repo_1.index.add(["file7.txt"])
        remote_repo_1.index.commit(
            "Update file7.txt",
            commit_date=datetime.datetime(
                2024, 1, 1, tzinfo=datetime.timezone.utc,
            ),
        )

        remote_2_1 = remote_repo_2.create_remote("remote-1", remote_dir_1)
        remote_2_1.fetch(["master"])
        remote_2_master = remote_repo_2.create_head(
            "master",
            remote_2_1.refs["master"],
        )

        remote_2_branch_3 = remote_repo_2.create_head(
            "branch-3", remote_2_master.commit
        )
        remote_repo_2.head.reference = remote_2_branch_3
        remote_repo_2.head.reset(index=True, working_tree=True)
        write_file(remote_repo_2, "file3.txt", "File 3 modified")
        remote_repo_2.index.add(["file3.txt"])
        remote_repo_2.index.commit(
            "Update file3.txt",
            commit_date=datetime.datetime(
                2024, 1, 1, tzinfo=datetime.timezone.utc,
            ),
        )

        remote_2_branch_4 = remote_repo_2.create_head(
            "branch-4", remote_2_master.commit
        )
        remote_repo_2.head.reference = remote_2_branch_4
        remote_repo_2.head.reset(index=True, working_tree=True)
        write_file(remote_repo_2, "file4.txt", "File 4 modified")
        remote_repo_2.index.add(["file4.txt"])
        remote_repo_2.index.commit(
            "Update file4.txt",
            commit_date=datetime.datetime(
                2025, 1, 1, tzinfo=datetime.timezone.utc,
            ),
        )

        remote_2_branch_5 = remote_repo_2.create_head(
            "branch-5", remote_2_master.commit
        )
        remote_repo_2.head.reference = remote_2_branch_5
        remote_repo_2.head.reset(index=True, working_tree=True)
        write_file(remote_repo_2, "file5.txt", "File 5 modified")
        remote_repo_2.index.add(["file5.txt"])
        remote_repo_2.index.commit("Update file5.txt")

        with mock_target_branch(None):
            assert copyright.get_target_branch_upstream_commit(
                git_repo, Mock(),
            ) is None

        with mock_target_branch("branch-1"):
            assert copyright.get_target_branch_upstream_commit(
                git_repo, Mock(),
            ) is None

        remote_1 = git_repo.create_remote(
            "unconventional/remote/name/1", remote_dir_1,
        )
        remote_1.fetch([
            "master",
            "branch-1-renamed",
            "branch-2",
            "branch-3",
            "branch-4",
            "branch-7",
        ])
        remote_2 = git_repo.create_remote(
            "unconventional/remote/name/2", remote_dir_2,
        )
        remote_2.fetch(["branch-3", "branch-4", "branch-5"])

        main = git_repo.create_head("main", remote_1.refs["master"])

        branch_1 = git_repo.create_head("branch-1", remote_1.refs["master"])
        with branch_1.config_writer() as w:
            w.set_value("remote", "unconventional/remote/name/1")
            w.set_value("merge", "branch-1-renamed")
        git_repo.head.reference = branch_1
        git_repo.head.reset(index=True, working_tree=True)
        git_repo.index.remove("file1.txt", working_tree=True)
        git_repo.index.commit(
            "Remove file1.txt",
            commit_date=datetime.datetime(
                2024, 1, 1, tzinfo=datetime.timezone.utc,
            ),
        )

        branch_6 = git_repo.create_head("branch-6", remote_1.refs["master"])
        git_repo.head.reference = branch_6
        git_repo.head.reset(index=True, working_tree=True)
        git_repo.index.remove(["file6.txt"], working_tree=True)
        git_repo.index.commit("Remove file6.txt")

        branch_7 = git_repo.create_head("branch-7", remote_1.refs["master"])
        with branch_7.config_writer() as w:
            w.set_value("remote", "unconventional/remote/name/1")
            w.set_value("merge", "branch-7")
        git_repo.head.reference = branch_7
        git_repo.head.reset(index=True, working_tree=True)
        git_repo.index.remove(["file7.txt"], working_tree=True)
        git_repo.index.commit(
            "Remove file7.txt",
            commit_date=datetime.datetime(
                2024, 2, 1, tzinfo=datetime.timezone.utc,
            ),
        )

        git_repo.head.reference = main
        git_repo.head.reset(index=True, working_tree=True)

        with mock_target_branch("branch-1"):
            assert (
                copyright.get_target_branch_upstream_commit(git_repo, Mock())
                == remote_1.refs["branch-1-renamed"].commit
            )

        with mock_target_branch("branch-2"):
            assert (
                copyright.get_target_branch_upstream_commit(git_repo, Mock())
                == remote_1.refs["branch-2"].commit
            )

        with mock_target_branch("branch-3"):
            assert (
                copyright.get_target_branch_upstream_commit(git_repo, Mock())
                == remote_1.refs["branch-3"].commit
            )

        with mock_target_branch("branch-4"):
            assert (
                copyright.get_target_branch_upstream_commit(git_repo, Mock())
                == remote_2.refs["branch-4"].commit
            )

        with mock_target_branch("branch-5"):
            assert (
                copyright.get_target_branch_upstream_commit(git_repo, Mock())
                == remote_2.refs["branch-5"].commit
            )

        with mock_target_branch("branch-6"):
            assert (
                copyright.get_target_branch_upstream_commit(git_repo, Mock())
                == branch_6.commit
            )

        with mock_target_branch("branch-7"):
            assert (
                copyright.get_target_branch_upstream_commit(git_repo, Mock())
                == branch_7.commit
            )

        with mock_target_branch("nonexistent-branch"):
            assert (
                copyright.get_target_branch_upstream_commit(git_repo, Mock())
                == main.commit
            )

        with mock_target_branch(None):
            assert (
                copyright.get_target_branch_upstream_commit(git_repo, Mock())
                == main.commit
            )


def test_get_changed_files(git_repo):
    def mock_os_walk(top):
        return patch(
            "os.walk",
            Mock(
                return_value=(
                    (
                        (
                            "."
                            if (rel := os.path.relpath(dirpath, top)) == "."
                            else os.path.join(".", rel)
                        ),
                        dirnames,
                        filenames,
                    )
                    for dirpath, dirnames, filenames in os.walk(top)
                )
            ),
        )

    with (
        tempfile.TemporaryDirectory() as non_git_dir,
        patch("os.getcwd", Mock(return_value=non_git_dir)),
        mock_os_walk(non_git_dir),
    ):
        with open(os.path.join(non_git_dir, "top.txt"), "w") as f:
            f.write("Top file\n")
        os.mkdir(os.path.join(non_git_dir, "subdir1"))
        os.mkdir(os.path.join(non_git_dir, "subdir1/subdir2"))
        with open(
            os.path.join(non_git_dir, "subdir1", "subdir2", "sub.txt"), "w"
        ) as f:
            f.write("Subdir file\n")
        assert copyright.get_changed_files(Mock()) == {
            "top.txt": ("A", None),
            "subdir1/subdir2/sub.txt": ("A", None),
        }

    def fn(filename):
        return os.path.join(git_repo.working_tree_dir, filename)

    def write_file(filename, contents):
        with open(fn(filename), "w") as f:
            f.write(contents)

    def file_contents(verbed):
        return f"This file will be {verbed}\n" * 100

    write_file("untouched.txt", file_contents("untouched"))
    write_file("copied.txt", file_contents("copied"))
    write_file("modified_and_copied.txt", file_contents("modified and copied"))
    write_file("copied_and_modified.txt", file_contents("copied and modified"))
    write_file("deleted.txt", file_contents("deleted"))
    write_file("renamed.txt", file_contents("renamed"))
    write_file(
        "modified_and_renamed.txt", file_contents("modified and renamed")
    )
    write_file("modified.txt", file_contents("modified"))
    write_file("chmodded.txt", file_contents("chmodded"))
    write_file("untracked.txt", file_contents("untracked"))
    git_repo.index.add(
        [
            "untouched.txt",
            "copied.txt",
            "modified_and_copied.txt",
            "copied_and_modified.txt",
            "deleted.txt",
            "renamed.txt",
            "modified_and_renamed.txt",
            "modified.txt",
            "chmodded.txt",
        ]
    )

    with (
        patch("os.getcwd", Mock(return_value=git_repo.working_tree_dir)),
        mock_os_walk(git_repo.working_tree_dir),
        patch(
            "rapids_pre_commit_hooks.copyright."
            "get_target_branch_upstream_commit",
            Mock(return_value=None),
        ),
    ):
        assert copyright.get_changed_files(Mock()) == {
            "untouched.txt": ("A", None),
            "copied.txt": ("A", None),
            "modified_and_copied.txt": ("A", None),
            "copied_and_modified.txt": ("A", None),
            "deleted.txt": ("A", None),
            "renamed.txt": ("A", None),
            "modified_and_renamed.txt": ("A", None),
            "modified.txt": ("A", None),
            "chmodded.txt": ("A", None),
            "untracked.txt": ("A", None),
        }

    git_repo.index.commit("Initial commit")

    # Ensure that diff is done against merge base, not branch tip
    git_repo.index.remove(["modified.txt"], working_tree=True)
    git_repo.index.commit("Remove modified.txt")

    pr_branch = git_repo.create_head("pr", "HEAD~")
    git_repo.head.reference = pr_branch
    git_repo.head.reset(index=True, working_tree=True)

    write_file("copied_2.txt", file_contents("copied"))
    git_repo.index.remove(
        ["deleted.txt", "modified_and_renamed.txt"], working_tree=True
    )
    git_repo.index.move(["renamed.txt", "renamed_2.txt"])
    write_file(
        "modified.txt",
        file_contents("modified") + "This file has been modified\n",
    )
    os.chmod(fn("chmodded.txt"), 0o755)
    write_file("untouched.txt", file_contents("untouched") + "Oops\n")
    write_file("added.txt", file_contents("added"))
    write_file("added_and_deleted.txt", file_contents("added and deleted"))
    write_file(
        "modified_and_copied.txt",
        file_contents("modified and copied") + "This file has been modified\n",
    )
    write_file(
        "modified_and_copied_2.txt", file_contents("modified and copied")
    )
    write_file(
        "copied_and_modified_2.txt",
        file_contents("copied and modified") + "This file has been modified\n",
    )
    write_file(
        "modified_and_renamed_2.txt",
        file_contents("modified and renamed")
        + "This file has been modified\n",
    )
    git_repo.index.add(
        [
            "untouched.txt",
            "added.txt",
            "added_and_deleted.txt",
            "modified_and_copied.txt",
            "modified_and_copied_2.txt",
            "copied_and_modified_2.txt",
            "copied_2.txt",
            "modified_and_renamed_2.txt",
            "modified.txt",
            "chmodded.txt",
        ]
    )
    write_file("untouched.txt", file_contents("untouched"))
    os.unlink(fn("added_and_deleted.txt"))

    target_branch = git_repo.heads["master"]
    merge_base = git_repo.merge_base(target_branch, "HEAD")[0]
    old_files = {
        blob.path: blob
        for blob in merge_base.tree.traverse(
            lambda b, _: isinstance(b, git.Blob)
        )
    }

    # Truly need to be checked
    changed = {
        "added.txt": ("A", None),
        "untracked.txt": ("A", None),
        "modified_and_renamed_2.txt": ("R", "modified_and_renamed.txt"),
        "modified.txt": ("M", "modified.txt"),
        "copied_and_modified_2.txt": ("C", "copied_and_modified.txt"),
        "modified_and_copied.txt": ("M", "modified_and_copied.txt"),
    }

    # Superfluous, but harmless because the content is identical
    superfluous = {
        "chmodded.txt": ("M", "chmodded.txt"),
        "modified_and_copied_2.txt": ("C", "modified_and_copied.txt"),
        "copied_2.txt": ("C", "copied.txt"),
        "renamed_2.txt": ("R", "renamed.txt"),
    }

    with (
        patch("os.getcwd", Mock(return_value=git_repo.working_tree_dir)),
        patch(
            "rapids_pre_commit_hooks.copyright."
            "get_target_branch_upstream_commit",
            Mock(return_value=target_branch.commit),
        ),
    ):
        changed_files = copyright.get_changed_files(Mock())
    assert {
        path: (change_type, old_blob.path if old_blob else None)
        for path, (change_type, old_blob) in changed_files.items()
    } == changed | superfluous

    for new, (_, old) in changed.items():
        if old:
            with open(fn(new), "rb") as f:
                new_contents = f.read()
            old_contents = old_files[old].data_stream.read()
            assert new_contents != old_contents
            assert changed_files[new][1].data_stream.read() == old_contents

    for new, (_, old) in superfluous.items():
        if old:
            with open(fn(new), "rb") as f:
                new_contents = f.read()
            old_contents = old_files[old].data_stream.read()
            assert new_contents == old_contents
            assert changed_files[new][1].data_stream.read() == old_contents


def test_get_changed_files_multiple_merge_bases(git_repo):
    def fn(filename):
        return os.path.join(git_repo.working_tree_dir, filename)

    def write_file(filename, contents):
        with open(fn(filename), "w") as f:
            f.write(contents)

    write_file("file1.txt", "File 1\n")
    write_file("file2.txt", "File 2\n")
    write_file("file3.txt", "File 3\n")
    git_repo.index.add(["file1.txt", "file2.txt", "file3.txt"])
    git_repo.index.commit("Initial commit")

    branch_1 = git_repo.create_head("branch-1", "master")
    git_repo.head.reference = branch_1
    git_repo.index.reset(index=True, working_tree=True)
    write_file("file1.txt", "File 1 modified\n")
    git_repo.index.add("file1.txt")
    git_repo.index.commit(
        "Modify file1.txt",
        commit_date=datetime.datetime(
            2024, 1, 1, tzinfo=datetime.timezone.utc
        ),
    )

    branch_2 = git_repo.create_head("branch-2", "master")
    git_repo.head.reference = branch_2
    git_repo.index.reset(index=True, working_tree=True)
    write_file("file2.txt", "File 2 modified\n")
    git_repo.index.add("file2.txt")
    git_repo.index.commit(
        "Modify file2.txt",
        commit_date=datetime.datetime(
            2024, 2, 1, tzinfo=datetime.timezone.utc
        ),
    )

    branch_1_2 = git_repo.create_head("branch-1-2", "master")
    git_repo.head.reference = branch_1_2
    git_repo.index.reset(index=True, working_tree=True)
    write_file("file1.txt", "File 1 modified\n")
    write_file("file2.txt", "File 2 modified\n")
    git_repo.index.add(["file1.txt", "file2.txt"])
    git_repo.index.commit(
        "Merge branches branch-1 and branch-2",
        parent_commits=[branch_1.commit, branch_2.commit],
        commit_date=datetime.datetime(
            2024, 3, 1, tzinfo=datetime.timezone.utc
        ),
    )

    branch_3 = git_repo.create_head("branch-3", "master")
    git_repo.head.reference = branch_3
    git_repo.index.reset(index=True, working_tree=True)
    write_file("file1.txt", "File 1 modified\n")
    write_file("file2.txt", "File 2 modified\n")
    git_repo.index.add(["file1.txt", "file2.txt"])
    git_repo.index.commit(
        "Merge branches branch-1 and branch-2",
        parent_commits=[branch_1.commit, branch_2.commit],
        commit_date=datetime.datetime(
            2024, 4, 1, tzinfo=datetime.timezone.utc
        ),
    )
    write_file("file3.txt", "File 3 modified\n")
    git_repo.index.add("file3.txt")
    git_repo.index.commit(
        "Modify file3.txt",
        commit_date=datetime.datetime(
            2024, 5, 1, tzinfo=datetime.timezone.utc
        ),
    )

    with (
        patch("os.getcwd", Mock(return_value=git_repo.working_tree_dir)),
        patch(
            "rapids_pre_commit_hooks.copyright.get_target_branch",
            Mock(return_value="branch-1-2"),
        ),
    ):
        changed_files = copyright.get_changed_files(Mock())
    assert {
        path: (change_type, old_blob.path if old_blob else None)
        for path, (change_type, old_blob) in changed_files.items()
    } == {
        "file1.txt": ("M", "file1.txt"),
        "file2.txt": ("M", "file2.txt"),
        "file3.txt": ("M", "file3.txt"),
    }


@pytest.mark.parametrize(
    ["filename", "normalized_filename"],
    [
        (
            "file.txt",
            "file.txt",
        ),
        (
            "sub/file.txt",
            "sub/file.txt",
        ),
        (
            "sub//file.txt",
            "sub/file.txt",
        ),
        (
            "sub/../file.txt",
            "file.txt",
        ),
        (
            "./file.txt",
            "file.txt",
        ),
        (
            "../file.txt",
            None,
        ),
        (
            os.path.join(os.getcwd(), "file.txt"),
            "file.txt",
        ),
        (
            os.path.join("..", os.path.basename(os.getcwd()), "file.txt"),
            "file.txt",
        ),
    ],
)
def test_normalize_git_filename(filename, normalized_filename):
    assert copyright.normalize_git_filename(filename) == normalized_filename


@pytest.mark.parametrize(
    ["path", "present"],
    [
        ("top.txt", True),
        ("sub1/sub2/sub.txt", True),
        ("nonexistent.txt", False),
        ("nonexistent/sub.txt", False),
    ],
)
def test_find_blob(git_repo, path, present):
    with open(os.path.join(git_repo.working_tree_dir, "top.txt"), "w"):
        pass
    os.mkdir(os.path.join(git_repo.working_tree_dir, "sub1"))
    os.mkdir(os.path.join(git_repo.working_tree_dir, "sub1", "sub2"))
    with open(
        os.path.join(git_repo.working_tree_dir, "sub1", "sub2", "sub.txt"), "w"
    ):
        pass
    git_repo.index.add(["top.txt", "sub1/sub2/sub.txt"])
    git_repo.index.commit("Initial commit")

    blob = copyright.find_blob(git_repo.head.commit.tree, path)
    if present:
        assert blob.path == path
    else:
        assert blob is None


@freeze_time("2024-01-18")
@pytest.mark.parametrize(
    [
        "target_branch",
        "filename",
        "contents",
        "warning_context",
        "op",
        "old_filename",
        "old_contents",
        "force_spdx",
    ],
    [
        (
            "branch-1",
            "file1.txt",
            "File 1 modified",
            contextlib.nullcontext(),
            None,
            None,
            None,
            False,
        ),
        (
            "branch-1",
            "file5.txt",
            "File 2",
            contextlib.nullcontext(),
            "R",
            "dir/file2.txt",
            "File 2",
            False,
        ),
        (
            "branch-1",
            "file3.txt",
            "File 3 modified",
            contextlib.nullcontext(),
            "M",
            "file3.txt",
            "File 3",
            False,
        ),
        (
            "branch-1",
            "file4.txt",
            "File 4 modified",
            contextlib.nullcontext(),
            "M",
            "file4.txt",
            "File 4",
            False,
        ),
        (
            "branch-1",
            "file6.txt",
            "File 6",
            contextlib.nullcontext(),
            "A",
            None,
            None,
            False,
        ),
        (
            "branch-2",
            "file1.txt",
            "File 1 modified",
            contextlib.nullcontext(),
            "M",
            "file1.txt",
            "File 1",
            False,
        ),
        (
            "branch-2",
            "./file1.txt",
            "File 1 modified",
            contextlib.nullcontext(),
            "M",
            "file1.txt",
            "File 1",
            False,
        ),
        (
            "branch-2",
            "../file1.txt",
            "File 1 modified",
            pytest.warns(
                copyright.ConflictingFilesWarning,
                match=(
                    r'^File "\.\./file1\.txt" is outside of current '
                    r"directory\. Not running linter on it\.$"
                ),
            ),
            None,
            None,
            None,
            False,
        ),
        (
            "branch-2",
            "file5.txt",
            "File 2",
            contextlib.nullcontext(),
            "R",
            "dir/file2.txt",
            "File 2",
            False,
        ),
        (
            "branch-2",
            "file3.txt",
            "File 3 modified",
            contextlib.nullcontext(),
            "M",
            "file3.txt",
            "File 3",
            False,
        ),
        (
            "branch-2",
            "file4.txt",
            "File 4 modified",
            contextlib.nullcontext(),
            "M",
            "file4.txt",
            "File 4",
            False,
        ),
        (
            "branch-2",
            "file6.txt",
            "File 6",
            contextlib.nullcontext(),
            "A",
            None,
            None,
            False,
        ),
        (
            "branch-1",
            "file1.txt",
            "File 1 modified",
            contextlib.nullcontext(),
            "M",
            "file1.txt",
            "File 1 modified",
            True,
        ),
    ],
)
def test_check_copyright(
    git_repo,
    target_branch,
    filename,
    contents,
    warning_context,
    op,
    old_filename,
    old_contents,
    force_spdx,
):
    def fn(filename):
        return os.path.join(git_repo.working_tree_dir, filename)

    def write_file(filename, contents):
        with open(fn(filename), "w") as f:
            f.write(contents)

    def file_contents(contents):
        return dedent(
            f"""\
            Copyright (c) 2021-2023 NVIDIA CORPORATION
            {contents}
            """
        )

    os.mkdir(os.path.join(git_repo.working_tree_dir, "dir"))
    write_file("file1.txt", file_contents("File 1"))
    write_file("dir/file2.txt", file_contents("File 2"))
    write_file("file3.txt", file_contents("File 3"))
    write_file("file4.txt", file_contents("File 4"))
    git_repo.index.add(
        ["file1.txt", "dir/file2.txt", "file3.txt", "file4.txt"]
    )
    git_repo.index.commit("Initial commit")

    branch_1 = git_repo.create_head("branch-1", "master")
    git_repo.head.reference = branch_1
    git_repo.head.reset(index=True, working_tree=True)
    write_file("file1.txt", file_contents("File 1 modified"))
    git_repo.index.add(["file1.txt"])
    git_repo.index.commit("Update file1.txt")

    branch_2 = git_repo.create_head("branch-2", "master")
    git_repo.head.reference = branch_2
    git_repo.head.reset(index=True, working_tree=True)
    write_file("dir/file2.txt", file_contents("File 2 modified"))
    git_repo.index.add(["dir/file2.txt"])
    git_repo.index.commit("Update file2.txt")

    pr = git_repo.create_head("pr", "branch-1")
    git_repo.head.reference = pr
    git_repo.head.reset(index=True, working_tree=True)
    write_file("file3.txt", file_contents("File 3 modified"))
    git_repo.index.add(["file3.txt"])
    git_repo.index.commit("Update file3.txt")
    write_file("file4.txt", file_contents("File 4 modified"))
    git_repo.index.add(["file4.txt"])
    git_repo.index.commit("Update file4.txt")
    git_repo.index.move(["dir/file2.txt", "file5.txt"])
    git_repo.index.commit("Rename file2.txt to file5.txt")

    write_file("file6.txt", file_contents("File 6"))

    def mock_repo_cwd():
        return patch("os.getcwd", Mock(return_value=git_repo.working_tree_dir))

    def mock_target_branch_upstream_commit(target_branch):
        def func(repo, args):
            assert target_branch == args.target_branch
            return repo.heads[target_branch].commit

        return patch(
            "rapids_pre_commit_hooks.copyright."
            "get_target_branch_upstream_commit",
            func,
        )

    def mock_apply_copyright_check():
        return patch(
            "rapids_pre_commit_hooks.copyright.apply_copyright_check", Mock()
        )

    mock_args = Mock(
        target_branch=target_branch,
        batch=False,
        spdx=False,
        force_spdx=force_spdx,
    )

    with mock_repo_cwd(), mock_target_branch_upstream_commit(target_branch):
        copyright_checker = copyright.check_copyright(mock_args)

    linter = Linter(filename, file_contents(contents))
    with mock_apply_copyright_check() as apply_copyright_check:
        with warning_context:
            copyright_checker(linter, mock_args)
        if op is None:
            apply_copyright_check.assert_not_called()
        else:
            apply_copyright_check.assert_called_once_with(
                linter,
                mock_args,
                op,
                old_filename,
                None if old_contents is None else file_contents(old_contents),
            )
