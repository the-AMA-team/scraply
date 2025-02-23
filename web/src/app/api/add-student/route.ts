import { NextRequest, NextResponse } from "next/server";
import { PrismaClient } from '@prisma/client';
import prisma from '../../../util/connect'

// code for updating the classrooms
export const POST = async (req: NextRequest) => {
  try {
    const { teacherEmail, student } = await req.json();

    console.log(teacherEmail, student);

    // const updatedAttempt = await prisma.attempt.updateMany({
    //     where: {
    //         userId: "004",
    //         id: "672c3d5ac11adb72d492e34c"
    //     },
    //     data:  {"level":"L1","rating":-1,"archetecture":"","lastLoss":-1}
    // });

    const classroom = await prisma.classrooms.findFirst({
        where: {
            teacher: teacherEmail,
            // classroomName: classroomName
        }
    })

    if (!classroom) {
        return NextResponse.json({ error: "Not Found" }, { status: 404 });
    }

    console.log(classroom);

    let curStudents = classroom.students 
    curStudents.push(student)

    const addStudent = await prisma.classrooms.updateMany({
        where: {
            teacher: teacherEmail,
        },
        data: {
            students: curStudents
        }
    })

    if (addStudent.count == 0) {
      return NextResponse.json({ error: "Not Found" }, { status: 404 });
    }
    return NextResponse.json(
      { updatedClassroom: addStudent },
      { status: 200 }
    );
  } catch (error) {
    // console.error(error);
    return NextResponse.json({ error }, { status: 500 });
  }
};
