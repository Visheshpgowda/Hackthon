import express from 'express';
import { addMarks } from '../controllers/studentControllers/marks.controllers.js';
import { addStudent } from '../controllers/studentControllers/addStudent.controllers.js';
import { getStudentListByClass } from '../controllers/studentControllers/studentListByClass.controllers.js';
import { addStudentAttendance } from '../controllers/studentControllers/studentattd.controllers.js';
const router = express.Router();


router.post('/addStudent', addStudent); 
router.post('/addMarks', addMarks);
router.post("/getstudents",getStudentListByClass)
router.post("/addattendance",addStudentAttendance)
export default router;