import unittest

from app import server


class CancelEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = server.app.test_client()
        with server.JOBS_LOCK:
            server.JOBS.clear()

    def tearDown(self):
        with server.JOBS_LOCK:
            server.JOBS.clear()

    def test_running_job_can_be_cancelled(self):
        with server.JOBS_LOCK:
            server.JOBS["job"] = {"status": "running", "cancel_requested": False}
        response = self.client.post("/api/job/job/cancel")
        self.assertEqual(response.status_code, 200)
        with server.JOBS_LOCK:
            self.assertTrue(server.JOBS["job"]["cancel_requested"])

    def test_terminal_job_rejects_late_cancel(self):
        for status in ("done", "error", "cancelled"):
            with self.subTest(status=status):
                with server.JOBS_LOCK:
                    server.JOBS["job"] = {
                        "status": status,
                        "cancel_requested": False,
                    }
                response = self.client.post("/api/job/job/cancel")
                self.assertEqual(response.status_code, 400)
                self.assertIn(status, response.get_json()["error"])

    def test_unknown_job_returns_not_found(self):
        response = self.client.post("/api/job/missing/cancel")
        self.assertEqual(response.status_code, 404)

    def test_checkpoint_raises_after_cancel_request(self):
        with server.JOBS_LOCK:
            server.JOBS["job"] = {"status": "running", "cancel_requested": True}
        with self.assertRaises(server.JobCancelled):
            server.check_cancelled("job")


if __name__ == "__main__":
    unittest.main()
