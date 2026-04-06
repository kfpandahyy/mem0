import json
import pytest
from unittest.mock import Mock, patch

from mem0.vector_stores.gaussdb import GaussDB, OutputData


@pytest.fixture
def mock_connection_pool():
    pool = Mock()
    conn = Mock()
    cursor = Mock()

    cursor.fetchall = Mock(return_value=[])
    cursor.fetchone = Mock(return_value=None)
    cursor.execute = Mock()
    cursor.__enter__ = Mock(return_value=cursor)
    cursor.__exit__ = Mock(return_value=False)

    conn.cursor = Mock(return_value=cursor)
    conn.commit = Mock()
    conn.rollback = Mock()
    conn.__enter__ = Mock(return_value=conn)
    conn.__exit__ = Mock(return_value=False)

    pool.connection = Mock(return_value=conn)
    pool.close = Mock()

    return pool, conn, cursor


@pytest.fixture
def gaussdb_instance(mock_connection_pool):
    pool, conn, cursor = mock_connection_pool
    with patch("mem0.vector_stores.gaussdb.ConnectionPool") as MockPool:
        MockPool.return_value = pool
        instance = GaussDB(
            host="localhost",
            port=5432,
            user="test",
            password="test",
            dbname="testdb",
            collection_name="test_col",
            embedding_model_dims=4,
        )
        instance.connection_pool = pool
        return instance


def test_init_creates_pool(mock_connection_pool):
    pool, conn, cursor = mock_connection_pool
    with patch("mem0.vector_stores.gaussdb.ConnectionPool") as MockPool:
        MockPool.return_value = pool
        GaussDB(
            host="localhost", port=5432, user="test", password="test",
            dbname="testdb", collection_name="test_col", embedding_model_dims=4,
        )
        call_kwargs = MockPool.call_args
        conninfo = call_kwargs[0][0]
        assert "localhost" in conninfo
        assert "5432" in conninfo
        assert "test" in conninfo
        assert "testdb" in conninfo


def test_init_with_connection_pool(mock_connection_pool):
    pool, conn, cursor = mock_connection_pool
    with patch("mem0.vector_stores.gaussdb.ConnectionPool") as MockPool:
        GaussDB(
            host="localhost", port=5432, user="test", password="test",
            dbname="testdb", collection_name="test_col", embedding_model_dims=4,
            connection_pool=pool,
        )
        MockPool.assert_not_called()


def test_init_creates_collection_if_not_exists(mock_connection_pool):
    pool, conn, cursor = mock_connection_pool
    cursor.fetchall.return_value = []
    with patch("mem0.vector_stores.gaussdb.ConnectionPool") as MockPool:
        MockPool.return_value = pool
        GaussDB(
            host="localhost", port=5432, user="test", password="test",
            dbname="testdb", collection_name="test_col", embedding_model_dims=4,
        )
    sql_calls = " ".join(str(c) for c in cursor.execute.call_args_list)
    assert "CREATE TABLE" in sql_calls


def test_init_skips_create_if_exists(mock_connection_pool):
    pool, conn, cursor = mock_connection_pool
    cursor.fetchall.return_value = [{"table_name": "test_col"}]
    with patch("mem0.vector_stores.gaussdb.ConnectionPool") as MockPool:
        MockPool.return_value = pool
        cursor.execute.reset_mock()
        GaussDB(
            host="localhost", port=5432, user="test", password="test",
            dbname="testdb", collection_name="test_col", embedding_model_dims=4,
        )
    sql_calls = " ".join(str(c) for c in cursor.execute.call_args_list)
    assert "CREATE TABLE" not in sql_calls


def test_create_col_without_hnsw(gaussdb_instance, mock_connection_pool):
    pool, conn, cursor = mock_connection_pool
    cursor.execute.reset_mock()
    gaussdb_instance.create_col(name="new_col", vector_size=4)
    sql_calls = " ".join(str(c) for c in cursor.execute.call_args_list)
    assert "CREATE TABLE" in sql_calls
    assert "CREATE INDEX" not in sql_calls


def test_create_col_with_hnsw(mock_connection_pool):
    pool, conn, cursor = mock_connection_pool
    cursor.fetchall.return_value = [{"table_name": "test_col"}]
    with patch("mem0.vector_stores.gaussdb.ConnectionPool") as MockPool:
        MockPool.return_value = pool
        instance = GaussDB(
            host="localhost", port=5432, user="test", password="test",
            dbname="testdb", collection_name="test_col", embedding_model_dims=4,
            hnsw=True,
        )
        instance.connection_pool = pool
    cursor.execute.reset_mock()
    instance.create_col(name="new_col", vector_size=4)
    sql_calls = " ".join(str(c) for c in cursor.execute.call_args_list)
    assert "hnsw" in sql_calls.lower()


def test_insert_executes_for_each_vector(gaussdb_instance, mock_connection_pool):
    pool, conn, cursor = mock_connection_pool
    cursor.execute.reset_mock()
    gaussdb_instance.insert(
        vectors=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
        payloads=[{"text": "a"}, {"text": "b"}],
        ids=["id1", "id2"],
    )
    assert cursor.execute.call_count == 2


def test_insert_uses_on_duplicate_key_update(gaussdb_instance, mock_connection_pool):
    pool, conn, cursor = mock_connection_pool
    cursor.execute.reset_mock()
    gaussdb_instance.insert(vectors=[[0.1, 0.2, 0.3, 0.4]], ids=["id1"])
    sql = cursor.execute.call_args[0][0]
    assert "ON DUPLICATE KEY UPDATE" in sql
    assert "ON CONFLICT" not in sql


def test_insert_generates_ids_if_none(gaussdb_instance, mock_connection_pool):
    pool, conn, cursor = mock_connection_pool
    cursor.execute.reset_mock()
    gaussdb_instance.insert(vectors=[[0.1, 0.2, 0.3, 0.4]])
    assert cursor.execute.call_count == 1
    args = cursor.execute.call_args[0][1]
    assert args[0] is not None  # auto-generated id


def test_search_uses_cosine_operator(gaussdb_instance, mock_connection_pool):
    pool, conn, cursor = mock_connection_pool
    cursor.fetchall.return_value = [
        {"id": "id1", "distance": 0.1, "payload": json.dumps({"text": "hello"})}
    ]
    results = gaussdb_instance.search(query="test", vectors=[0.1, 0.2, 0.3, 0.4])
    sql = cursor.execute.call_args[0][0]
    assert "<=>" in sql
    assert isinstance(results, list)
    assert isinstance(results[0], OutputData)


def test_search_with_filters(gaussdb_instance, mock_connection_pool):
    pool, conn, cursor = mock_connection_pool
    cursor.fetchall.return_value = []
    gaussdb_instance.search(query="test", vectors=[0.1, 0.2, 0.3, 0.4], filters={"user_id": "alice"})
    sql = cursor.execute.call_args[0][0]
    assert "payload->>" in sql


def test_search_returns_correct_score(gaussdb_instance, mock_connection_pool):
    pool, conn, cursor = mock_connection_pool
    cursor.fetchall.return_value = [
        {"id": "id1", "distance": 0.42, "payload": {"text": "hello"}}
    ]
    results = gaussdb_instance.search(query="test", vectors=[0.1, 0.2, 0.3, 0.4])
    assert results[0].score == pytest.approx(0.42)


def test_delete_executes_correct_sql(gaussdb_instance, mock_connection_pool):
    pool, conn, cursor = mock_connection_pool
    cursor.execute.reset_mock()
    gaussdb_instance.delete("test_id")
    sql = cursor.execute.call_args[0][0]
    assert "DELETE FROM" in sql
    assert cursor.execute.call_args[0][1] == ("test_id",)


def test_update_vector_only(gaussdb_instance, mock_connection_pool):
    pool, conn, cursor = mock_connection_pool
    cursor.execute.reset_mock()
    gaussdb_instance.update("test_id", vector=[0.1, 0.2, 0.3, 0.4])
    assert cursor.execute.call_count == 1
    sql = cursor.execute.call_args[0][0]
    assert "vector" in sql
    assert "::vector" in sql


def test_update_payload_only(gaussdb_instance, mock_connection_pool):
    pool, conn, cursor = mock_connection_pool
    cursor.execute.reset_mock()
    gaussdb_instance.update("test_id", payload={"text": "updated"})
    assert cursor.execute.call_count == 1
    sql = cursor.execute.call_args[0][0]
    assert "payload" in sql
    assert "vector" not in sql


def test_get_returns_output_data(gaussdb_instance, mock_connection_pool):
    pool, conn, cursor = mock_connection_pool
    cursor.fetchone.return_value = {"id": "test_id", "payload": json.dumps({"text": "hello"})}
    result = gaussdb_instance.get("test_id")
    assert isinstance(result, OutputData)
    assert result.id == "test_id"
    assert result.score is None


def test_get_returns_none_if_not_found(gaussdb_instance, mock_connection_pool):
    pool, conn, cursor = mock_connection_pool
    cursor.fetchone.return_value = None
    result = gaussdb_instance.get("missing_id")
    assert result is None


def test_list_cols_queries_information_schema(gaussdb_instance, mock_connection_pool):
    pool, conn, cursor = mock_connection_pool
    cursor.fetchall.return_value = [{"table_name": "test_col"}]
    cols = gaussdb_instance.list_cols()
    sql = cursor.execute.call_args[0][0]
    assert "information_schema" in sql
    assert "public" in sql
    assert cols == ["test_col"]


def test_delete_col_drops_table(gaussdb_instance, mock_connection_pool):
    pool, conn, cursor = mock_connection_pool
    cursor.execute.reset_mock()
    gaussdb_instance.delete_col()
    sql = cursor.execute.call_args[0][0]
    assert "DROP TABLE IF EXISTS" in sql
    assert "test_col" in sql


def test_col_info_returns_dict(gaussdb_instance, mock_connection_pool):
    pool, conn, cursor = mock_connection_pool
    cursor.fetchone.return_value = {"table_name": "test_col", "row_count": 10, "total_size": "16 kB"}
    info = gaussdb_instance.col_info()
    assert info["name"] == "test_col"
    assert info["count"] == 10
    assert "size" in info


def test_col_info_returns_empty_if_not_found(gaussdb_instance, mock_connection_pool):
    pool, conn, cursor = mock_connection_pool
    cursor.fetchone.return_value = None
    info = gaussdb_instance.col_info()
    assert info == {}


def test_list_with_filters(gaussdb_instance, mock_connection_pool):
    pool, conn, cursor = mock_connection_pool
    cursor.fetchall.return_value = []
    gaussdb_instance.list(filters={"user_id": "alice"})
    sql = cursor.execute.call_args[0][0]
    assert "payload->>" in sql


def test_reset_deletes_and_recreates(gaussdb_instance, mock_connection_pool):
    pool, conn, cursor = mock_connection_pool
    cursor.execute.reset_mock()
    gaussdb_instance.reset()
    sql_calls = " ".join(str(c) for c in cursor.execute.call_args_list)
    assert "DROP TABLE" in sql_calls
    assert "CREATE TABLE" in sql_calls


def test_del_closes_pool(gaussdb_instance, mock_connection_pool):
    pool, conn, cursor = mock_connection_pool
    gaussdb_instance.__del__()
    pool.close.assert_called_once()


def test_output_data_model():
    data = OutputData(id="test_id", score=0.95, payload={"text": "hello"})
    assert data.id == "test_id"
    assert data.score == 0.95
    assert data.payload == {"text": "hello"}

    empty = OutputData(id=None, score=None, payload=None)
    assert empty.id is None


def test_config_requires_password_without_dsn():
    from mem0.configs.vector_stores.gaussdb import GaussDBConfig
    with pytest.raises(ValueError):
        GaussDBConfig(host="h", user="u", dbname="d")


def test_config_accepts_connection_string():
    from mem0.configs.vector_stores.gaussdb import GaussDBConfig
    cfg = GaussDBConfig(connection_string="host=h user=u dbname=d password=p")
    assert cfg.connection_string is not None


def test_config_rejects_extra_fields():
    from mem0.configs.vector_stores.gaussdb import GaussDBConfig
    with pytest.raises(ValueError, match="Extra fields"):
        GaussDBConfig(host="h", user="u", dbname="d", password="p", unknown_field="x")
