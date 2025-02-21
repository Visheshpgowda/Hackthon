
import express from 'express';
import { AddfacultyAttendance } from '../controllers/facultyControllers/facultyattd.controller.js';
import { getAllFacultyAttendance } from '../controllers/facultyControllers/facultyattd.controller.js';
import { getfaculty } from '../controllers/facultyControllers/facultyattd.controller.js';
const router = express.Router();
router.get('/getfaculty', getfaculty);
router.get('/getAllFacultyAttendance', getAllFacultyAttendance);
router.post('/addAttendance', AddfacultyAttendance);

export default router;