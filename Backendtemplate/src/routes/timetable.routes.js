import express from 'express';
import { getTimetable } from '../controllers/timetablecontroller/timetablecontroller.js';
import { substituteTeacher } from '../controllers/timetablecontroller/timetablecontroller.js';
import { addTimetable } from '../controllers/timetablecontroller/timetablecontroller.js';

const router = express.Router();
router.post('/add',addTimetable);
router.get('/getTimetable', getTimetable);

router.put('/substituteTeacher', substituteTeacher);

export default router;